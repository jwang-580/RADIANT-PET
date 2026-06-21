# input lesion mask, SUV, and organ mask files
# output lesion description


import nibabel as nib
import numpy as np
import json
import os
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy import ndimage as ndi

# Import utility functions
from utils.lesion_features import (
    convert_numpy_types, LesionShape, connectivity_structure, bbox_from_mask,
    resample_mask_to_suv_grid, compute_shape_metrics,
    calculate_spatial_relationship,
    labels_by_contains, analyze_organ_overlaps, combine_organ_results, determine_vertebrae_level,
    precompute_nearest_organ_fields, find_closest_organs, precompute_vertebrae_levels
)

@dataclass
class Lesion:
    """Class to represent a lesion with its properties"""
    id: int
    coordinates: np.ndarray  # center coordinates [x, y, z] in voxel indices
    volume_voxels: int  # number of voxels
    max_suv: Optional[float]
    mean_suv: Optional[float]
    bounding_box: Tuple[slice, slice, slice]
    organ_overlaps: Dict[str, float]
    closest_organs: Dict[str, Dict[str, float]]
    vertebrae_level: Optional[List[str]] = None
    shape: Optional['LesionShape'] = None

class MaskAnalyzer:
    """Pipeline for analyzing existing lesion masks"""
    
    def __init__(self, lesion_mask_file: str, organ_mask_files: dict, mapping_file: str, suv_file: Optional[str] = None):
        """
        Initialize the analyzer
        
        Args:
            lesion_mask_file: Path to the lesion segmentation NIfTI file
            organ_mask_files: Dictionary of mask type to file path
            mapping_file: Path to the organ mapping JSON file
            suv_file: Optional path to the SUV NIfTI file for intensity stats
        """
        self.lesion_mask_file = lesion_mask_file
        self.organ_mask_files = organ_mask_files
        self.mapping_file = mapping_file
        self.suv_file = suv_file
        
        self.load_data()
        self.precompute_nearest_organ_fields()
        self.precompute_vertebrae_levels()
        
    def load_data(self):
        """Load all necessary images and metadata"""
        print(f"Loading lesion mask: {self.lesion_mask_file}")
        self.lesion_img = nib.load(self.lesion_mask_file)
        self.lesion_data = self.lesion_img.get_fdata().astype(int)
        self.shape = self.lesion_data.shape
        self.voxel_size = self.lesion_img.header.get_zooms()[:3]
        
        # Load SUV if provided
        if self.suv_file:
            print(f"Loading SUV file: {self.suv_file}")
            self.suv_img = nib.load(self.suv_file)
            self.suv_data = self.suv_img.get_fdata()
            
            # Check dimensions
            # Check dimensions and affine
            if (self.suv_data.shape != self.shape or 
                not np.allclose(self.suv_img.affine, self.lesion_img.affine, atol=1e-3)):
                
                print(f"  Note: SUV geometry differs from lesion mask.")
                print(f"  Creating resampled lesion mask aligned to SUV grid for intensity sampling...")
                
                # Resample lesion mask to SUV grid
                self.lesion_data_suv = resample_mask_to_suv_grid(
                    self.lesion_data, 
                    self.lesion_img.affine,
                    self.suv_img.affine, 
                    self.suv_data.shape
                )
            else:
                # Geometries match, use original data
                self.lesion_data_suv = self.lesion_data
        else:
            self.suv_data = None
            self.lesion_data_suv = None
            print("No SUV file provided. Intensity metrics will be skipped.")

        # Load organ mapping
        with open(self.mapping_file, 'r') as f:
            mapping_data = json.load(f)
            
            # Combine all organ mask types with unique label ranges
            self.organ_mapping = {}
            self.organ_types = {}
            self.label_remapping = {}
            
            current_label_offset = 0
            mask_type_offsets = {}
            
            for mask_type, organs in mapping_data.items():
                mask_type_offsets[mask_type] = current_label_offset
                self.label_remapping[mask_type] = {}
                
                for label_str, organ_name in organs.items():
                    original_label = int(label_str)
                    new_label = original_label + current_label_offset
                    self.organ_mapping[new_label] = organ_name
                    self.organ_types[new_label] = mask_type
                    self.label_remapping[mask_type][original_label] = new_label
                
                if organs:
                    max_original_label = max(int(label) for label in organs.keys())
                    current_label_offset += max_original_label + 1000
            
            print(f"Total loaded: {len(self.organ_mapping)} organs from {len(mapping_data)} mask types")

        # Load and combine organ masks
        self.organ_data = np.zeros(self.shape, dtype=int)
        
        for mask_type, mask_file in self.organ_mask_files.items():
            mask_img = nib.load(mask_file)
            mask_data = mask_img.get_fdata().astype(int)
            
            # Check if resampling is needed for this mask
            if not np.allclose(self.lesion_img.affine, mask_img.affine, atol=1e-3):
                print(f"  Warning: {mask_type} mask affine differs - resampling to lesion mask grid")
                mask_data = resample_mask_to_suv_grid(mask_data, mask_img.affine, 
                                                    self.lesion_img.affine, self.shape)
            
            # Remap labels
            remapped_mask_data = np.zeros_like(mask_data)
            if mask_type in self.label_remapping:
                for original_label, new_label in self.label_remapping[mask_type].items():
                    remapped_mask_data[mask_data == original_label] = new_label
            
            # Merge
            non_zero_mask = remapped_mask_data > 0
            new_label_mask = non_zero_mask & (self.organ_data == 0)
            self.organ_data[new_label_mask] = remapped_mask_data[new_label_mask]
            
        print(f"Combined organ mask shape: {self.organ_data.shape}")

    def precompute_nearest_organ_fields(self) -> None:
        self.nearest_organ_dist_voxels, self.nearest_organ_label = precompute_nearest_organ_fields(self.organ_data)
        self._nearest_fields_ready = True

    def _ensure_nearest_fields(self) -> None:
        if not hasattr(self, "_nearest_fields_ready") or not self._nearest_fields_ready:
            self.precompute_nearest_organ_fields()
            
    def precompute_vertebrae_levels(self):
        self.vertebrae_order, self.vertebrae_axial_ranges = precompute_vertebrae_levels(
            self.organ_data, self.organ_mapping
        )

    def analyze_lesions(self) -> List[Lesion]:
        """Analyze all lesions found in the mask"""
        lesion_ids = np.unique(self.lesion_data)
        lesion_ids = lesion_ids[lesion_ids > 0]
        
        print(f"Found {len(lesion_ids)} lesions in mask")
        
        lesions = []
        
        for lesion_id in lesion_ids:
            lesion_voxels = (self.lesion_data == lesion_id)
            volume_voxels = int(lesion_voxels.sum())
            
            # Coordinates
            coords = np.where(lesion_voxels)
            center_coords = np.array([np.mean(coords[0]), np.mean(coords[1]), np.mean(coords[2])])
            
            # Bounding Box
            min_coords = [int(np.min(c)) for c in coords]
            max_coords = [int(np.max(c)) for c in coords]
            bounding_box = tuple(slice(min_coords[i], max_coords[i] + 1) for i in range(3))
            
            # SUV Stats
            if self.suv_data is not None and self.lesion_data_suv is not None:
                # Use the SUV-aligned mask for intensity stats
                lesion_voxels_suv = (self.lesion_data_suv == lesion_id)
                
                if params_exist := np.any(lesion_voxels_suv):
                    max_suv = float(np.max(self.suv_data[lesion_voxels_suv]))
                    mean_suv = float(np.mean(self.suv_data[lesion_voxels_suv]))
                else:
                    # Lesion might be too small to appear in the (potentially lower-res) SUV grid
                    max_suv = None
                    mean_suv = None
            else:
                max_suv = None
                mean_suv = None
                
            # Organ Analysis
            organ_overlaps = analyze_organ_overlaps(self.organ_data, self.organ_mapping, lesion_voxels)
            self._ensure_nearest_fields()
            closest_organs = find_closest_organs(lesion_voxels, self.organ_data, self.organ_mapping,
                                               self.nearest_organ_dist_voxels, self.nearest_organ_label,
                                               self.shape, num_closest=5)
            combined_organs = combine_organ_results(organ_overlaps, closest_organs, target_count=3)
            
            # Vertebrae
            vertebrae_level = determine_vertebrae_level(lesion_voxels, self.vertebrae_axial_ranges)
            
            # Shape
            shape = compute_shape_metrics(lesion_voxels.astype(bool))
            
            lesion = Lesion(
                id=int(lesion_id),
                coordinates=center_coords,
                volume_voxels=volume_voxels,
                max_suv=max_suv,
                mean_suv=mean_suv,
                bounding_box=bounding_box,
                organ_overlaps=combined_organs['overlaps'],
                closest_organs=combined_organs['closest'],
                vertebrae_level=vertebrae_level,
                shape=shape
            )
            lesions.append(lesion)
            
        return lesions

def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze existing lesion segmentation mask")
    
    parser.add_argument("--lesion_mask", "-l", required=True, help="Path to lesion segmentation NIfTI file")
    parser.add_argument("--organ_mask_dir", "-o", default="data/", help="Directory containing organ mask NIfTI files")
    parser.add_argument("--organ_mask_prefix", required=True, help="Prefix for organ mask files")
    parser.add_argument("--mapping_file", "-m", default="data/totalsegmentator_index_mapping.json", help="Path to organ mapping JSON file")
    parser.add_argument("--suv_file", "-s", help="Optional path to SUV NIfTI file for intensity stats")
    parser.add_argument("--output_json", "-j", default="lesion_analysis.json", help="Output JSON file")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Validate inputs
    if not os.path.exists(args.lesion_mask):
        print(f"Error: Lesion mask not found: {args.lesion_mask}")
        return
        
    if not os.path.exists(args.mapping_file):
        print(f"Error: Mapping file not found: {args.mapping_file}")
        return

    # Load mapping to get mask types
    with open(args.mapping_file, 'r') as f:
        mapping_data = json.load(f)
        
    organ_mask_files = {}
    for mask_type in mapping_data.keys():
        mask_file_path = os.path.join(args.organ_mask_dir, f"{args.organ_mask_prefix}{mask_type}.nii")
        if os.path.exists(mask_file_path):
            organ_mask_files[mask_type] = mask_file_path
        else:
            print(f"Warning: {mask_type} mask file not found: {mask_file_path}")
            
    if not organ_mask_files:
        print("Error: No organ mask files found")
        return

    # Run Analysis
    analyzer = MaskAnalyzer(
        lesion_mask_file=args.lesion_mask,
        organ_mask_files=organ_mask_files,
        mapping_file=args.mapping_file,
        suv_file=args.suv_file
    )
    
    lesions = analyzer.analyze_lesions()
    
    # Format Output
    image_center_x = analyzer.shape[0] / 2.0
    image_center_y = analyzer.shape[1] / 2.0
    image_center_z = analyzer.shape[2] / 2.0
    
    lesion_data = []
    for i, lesion in enumerate(lesions, 1):
        # Center coordinates
        centered_x = float(lesion.coordinates[0] - image_center_x)
        centered_y = float(lesion.coordinates[1] - image_center_y)
        centered_z = float(lesion.coordinates[2] - image_center_z)
        
        lesion_dict = {
            'id': lesion.id,
            'center_coords': [centered_x, centered_y, centered_z],
            'volume_voxels': int(lesion.volume_voxels),
            'max_suv': float(lesion.max_suv) if lesion.max_suv is not None else None,
            'mean_suv': float(lesion.mean_suv) if lesion.mean_suv is not None else None,
            'organ_overlaps': {k: f"{float(v):.2f}%" for k, v in lesion.organ_overlaps.items()},
            'closest_organs': {k: {
                'distance_voxels': float(v['distance_voxels']),
                'spatial_relationship': v.get('spatial_relationship', {})
            } for k, v in lesion.closest_organs.items()},
            'vertebrae_level': lesion.vertebrae_level if lesion.vertebrae_level else []
        }
        
        if lesion.shape:
            lesion_dict['shape'] = {
                'elongation': float(lesion.shape.elongation),
                'flatness': float(lesion.shape.flatness),
                'solidity': float(lesion.shape.solidity),
                'surface_to_volume': float(lesion.shape.surface_to_volume),
                'max_diameter_voxels': float(lesion.shape.max_diameter_voxels),
                'surface_area_voxels': float(lesion.shape.surface_area_voxels)
            }
        lesion_data.append(lesion_dict)
        
    output_data = {
        'metadata': {
            'lesion_mask': args.lesion_mask,
            'suv_file': args.suv_file,
            'total_lesions': len(lesions),
            'image_shape': list(analyzer.shape),
            'voxel_size': list(analyzer.voxel_size)
        },
        'lesions': lesion_data
    }
    
    # Convert NumPy types to Python native types for JSON serialization
    output_data = convert_numpy_types(output_data)
    
    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print(f"Saved analysis of {len(lesions)} lesions to {args.output_json}")

if __name__ == "__main__":
    main()
