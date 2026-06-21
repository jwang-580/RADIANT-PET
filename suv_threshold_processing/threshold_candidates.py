import nibabel as nib
import numpy as np
import json
from scipy import ndimage
from scipy.ndimage import label
import os
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, distance
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, relabel_sequential
from skimage import morphology

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.lesion_features import (
    convert_numpy_types, LesionShape, connectivity_structure, bbox_from_mask,
    final_ccl_from_labelmap, filter_small_components, compute_shape_metrics,
    split_component_watershed, suv_based_watershed_large_component,
    resample_mask_to_suv_grid, compute_liver_suv, lesion_surface,
    spacing_zyx_from_voxel_size, calculate_spatial_relationship,
    labels_by_contains, create_exclusion_mask, create_z_boundary_exclusion_mask,
    analyze_organ_overlaps, combine_organ_results, determine_vertebrae_level,
    precompute_nearest_organ_fields, find_closest_organs, save_lesion_mask,
    precompute_vertebrae_levels
)

@dataclass
class Lesion:
    """Class to represent a lesion with its properties"""
    id: int
    coordinates: np.ndarray  # center coordinates [x, y, z] in voxel indices
    volume_voxels: int  # number of voxels
    max_suv: float
    mean_suv: float
    bounding_box: Tuple[slice, slice, slice]
    organ_overlaps: Dict[str, float]  # organ_name: overlap_percentage (combined with closest to total 3)
    closest_organs: Dict[str, Dict[str, float]]  # organ_name: {'distance_voxels': float} (combined with overlaps to total 3)
    vertebrae_level: Optional[List[str]] = None  # list of vertebrae names that the lesion spans
    shape: Optional['LesionShape'] = None  # shape metrics for the lesion

class PETLesionPipeline:
    """Pipeline for extracting lesions from PET scans and analyzing organ overlaps"""
    
    def __init__(self, suv_file: str, organ_mask_files: dict, mapping_file: str):
        """
        Initialize the pipeline with input files
        
        Args:
            suv_file: Path to the SUV NIfTI file
            organ_mask_files: Dictionary of mask type to file path (e.g., {'total': 'path1.nii', 'head_glands': 'path2.nii'})
            mapping_file: Path to the organ mapping JSON file (mask numbers to organ names)
        """
        self.suv_file = suv_file
        self.organ_mask_files = organ_mask_files
        self.mapping_file = mapping_file
        
        # Load organ mapping (all types) with label remapping to avoid conflicts
        with open(mapping_file, 'r') as f:
            mapping_data = json.load(f)
            
            # Combine all organ mask types with unique label ranges
            self.organ_mapping = {}
            self.organ_types = {}  # Track which type each organ belongs to
            self.label_remapping = {}  # Track original -> new label mapping per mask type
            
            current_label_offset = 0
            mask_type_offsets = {}
            
            for mask_type, organs in mapping_data.items():
                
                mask_type_offsets[mask_type] = current_label_offset
                self.label_remapping[mask_type] = {}
                
                for label_str, organ_name in organs.items():
                    original_label = int(label_str)
                    new_label = original_label + current_label_offset
                    
                    # Store the organ mapping with new label
                    self.organ_mapping[new_label] = organ_name
                    # Track which mask type this organ belongs to
                    self.organ_types[new_label] = mask_type
                    # Track the label remapping
                    self.label_remapping[mask_type][original_label] = new_label
                
                # Update offset for next mask type (use max label + 1000 for safety)
                if organs:
                    max_original_label = max(int(label) for label in organs.keys())
                    current_label_offset += max_original_label + 1000
            
            print(f"Total loaded: {len(self.organ_mapping)} organs from {len(mapping_data)} mask types")
            print(f"Label ranges: {mask_type_offsets}")
        
        # Define vertebrae range (mask numbers 25-50) 
        self.vertebrae_range = range(25, 51)
        
        self.load_images()
        self.precompute_nearest_organ_fields()
        self.precompute_vertebrae_levels()
    
    
    def load_images(self):
        """Load SUV and organ mask images"""
        self.suv_img = nib.load(self.suv_file)
        self.suv_data = self.suv_img.get_fdata()
        
        # Store voxel size from SUV image header for spacing calculations
        # voxel_size is [dim0, dim1, dim2] from NIfTI header
        self.voxel_size = self.suv_img.header.get_zooms()[:3]
        
        print(f"SUV data shape: {self.suv_data.shape}")
        
        # Load and combine all organ mask files
        self.organ_data = np.zeros(self.suv_data.shape, dtype=int)
        
        for mask_type, mask_file in self.organ_mask_files.items():
            mask_img = nib.load(mask_file)
            mask_data = mask_img.get_fdata().astype(int)
            
            # Check if resampling is needed for this mask
            if not np.allclose(self.suv_img.affine, mask_img.affine, atol=1e-3):
                print(f"  Warning: {mask_type} mask affine differs - resampling to SUV grid")
                mask_data = resample_mask_to_suv_grid(mask_data, mask_img.affine, 
                                                    self.suv_img.affine, self.suv_data.shape)
            
            # Remap labels to avoid conflicts between mask types
            remapped_mask_data = np.zeros_like(mask_data)
            if mask_type in self.label_remapping:
                for original_label, new_label in self.label_remapping[mask_type].items():
                    remapped_mask_data[mask_data == original_label] = new_label
            
            # Add this mask to the combined organ data (total, head_glands, etc.)
            # Only add non-zero values, handle overlaps by prioritizing existing labels
            non_zero_mask = remapped_mask_data > 0
            overlap_mask = non_zero_mask & (self.organ_data > 0)
            
            if np.any(overlap_mask):
                overlap_count = np.sum(overlap_mask)
                print(f"  Warning: {mask_type} mask has {overlap_count:,} overlapping voxels - keeping existing organ labels")
            
            # Add new labels where there's no existing organ data
            new_label_mask = non_zero_mask & (self.organ_data == 0)
            self.organ_data[new_label_mask] = remapped_mask_data[new_label_mask]
            
            unique_labels = np.unique(remapped_mask_data[non_zero_mask])
            added_labels = np.unique(remapped_mask_data[new_label_mask])
        
        print(f"Combined organ mask shape: {self.organ_data.shape}")
        unique_organs = np.unique(self.organ_data)
        unique_organs = unique_organs[unique_organs > 0]
        print(f"Total unique organ labels: {len(unique_organs)}")
        
        
        # Store the first mask image for affine reference
        self.organ_img = nib.load(list(self.organ_mask_files.values())[0])

    def _resample_organ_to_suv_grid(self):
        """
        Resample organ mask to match SUV voxel grid for 1:1 voxel correspondence.
        Uses proper coordinate transformation via RAS space.
        """
        print("Resampling organ mask to SUV voxel grid...")
        
        # Get affine matrices and array info
        suv_affine = self.suv_img.affine
        organ_affine = self.organ_img.affine
        suv_shape = self.suv_data.shape
        organ_shape = self.organ_data.shape
        
        # Create coordinate grids for the SUV space
        # We want to find what organ voxel corresponds to each SUV voxel
        i, j, k = np.mgrid[0:suv_shape[0], 0:suv_shape[1], 0:suv_shape[2]]
        
        # Convert SUV voxel coordinates to homogeneous coordinates
        suv_voxel_coords = np.column_stack([
            i.ravel(), j.ravel(), k.ravel(), np.ones(i.size)
        ])
        
        # Transform SUV voxels to RAS world coordinates
        ras_coords = suv_affine @ suv_voxel_coords.T
        
        # Transform RAS coordinates to organ voxel coordinates
        organ_affine_inv = np.linalg.inv(organ_affine)
        organ_voxel_coords = organ_affine_inv @ ras_coords
        
        # Extract just the spatial coordinates (drop homogeneous coordinate)
        organ_coords = organ_voxel_coords[:3, :].T
        organ_coords = organ_coords.reshape(suv_shape + (3,))
        
        # Use map_coordinates to sample organ data at the computed coordinates
        from scipy.ndimage import map_coordinates
        
        resampled_organ = map_coordinates(
            self.organ_data.astype(float),
            [organ_coords[..., 0], organ_coords[..., 1], organ_coords[..., 2]],
            order=0,  # Nearest neighbor for label data
            mode='constant',
            cval=0,  # Background value
            prefilter=False
        )
        
        # Update organ data and image
        self.organ_data = resampled_organ.astype(int)
        self.organ_img = nib.Nifti1Image(self.organ_data, suv_affine, self.suv_img.header)
        
        print(f"Organ mask resampled to SUV grid. New shape: {self.organ_data.shape}")



    def precompute_nearest_organ_fields(self) -> None:
        """
        Precompute per-voxel:
        - self.nearest_organ_dist_voxels: distance (voxels) to nearest organ voxel (label>0)
        - self.nearest_organ_label:   label of that nearest organ voxel
        """
        self.nearest_organ_dist_voxels, self.nearest_organ_label = precompute_nearest_organ_fields(self.organ_data)
        self._nearest_fields_ready = True

    def _ensure_nearest_fields(self) -> None:
        if not hasattr(self, "_nearest_fields_ready") or not self._nearest_fields_ready:
            self.precompute_nearest_organ_fields()
    
    def precompute_vertebrae_levels(self):
        """Pre-calculate vertebrae axial coordinate ranges for lesion matching"""
        self.vertebrae_order, self.vertebrae_axial_ranges = precompute_vertebrae_levels(
            self.organ_data, self.organ_mapping
        )


    def extract_lesions(self,
                        suv_threshold: float = 2.5,
                        grow_suv_threshold: float = 2.5,
                        min_voxels: int = 20,
                        connectivity: int = 18,
                        pre_min_voxels: int = 10,
                        post_min_voxels: Optional[int] = None,
                        exclude_brain: bool = False,
                        exclude_kidneys: bool = False,
                        exclude_bladder: bool = False,
                        z_boundary_voxels: int = 5,
) -> List["Lesion"]:
        """
        threshold → pre-detection Z-boundary masking → initial CCL → (PRE-FILTER tiny blobs) 
        → watershed split per component → (POST-FILTER tiny pieces) 
        → post-watershed exclusion filtering → final CCL → remove suvmax < liver suvmean → build Lesion objects (+ shape).
        
        Args:
            suv_threshold: Minimum SUV value for lesion detection
            min_voxels: Minimum number of voxels for lesion detection
            connectivity: Connectivity for connected component labeling (6, 18, or 26)
            pre_min_voxels: Minimum voxels for pre-filtering tiny initial blobs
            post_min_voxels: Minimum voxels for post-filtering watershed pieces
            exclude_brain: Whether to exclude brain from lesion detection
            exclude_kidneys: Whether to exclude kidneys from lesion detection
            exclude_bladder: Whether to exclude bladder from lesion detection
            z_boundary_voxels: Number of voxels to exclude from Z-axis boundaries
            
        Returns:
            List of extracted lesions
        """
        # working in voxel space only

        # 1) Create permissive regions by growing high-SUV seeds through all
        # connected voxels above the lower inclusion threshold.  With equal
        # thresholds this reduces to ordinary thresholding.
        seed_mask = self.suv_data >= suv_threshold
        growable_mask = self.suv_data >= grow_suv_threshold
        coarse_mask = ndi.binary_propagation(
            seed_mask,
            structure=connectivity_structure(connectivity),
            mask=growable_mask,
        )
        
        # 1.5) Create exclusion mask for organs with physiological uptake
        exclusion_mask = create_exclusion_mask(self.organ_data, self.organ_mapping,
                                             exclude_brain, exclude_kidneys, exclude_bladder, dilate_voxels=2)
        
        # 1.6) Add Z-axis boundary exclusion to remove edge artifacts
        z_boundary_mask = create_z_boundary_exclusion_mask(self.suv_data.shape, z_boundary_voxels)
        
        # Apply exclusions for pre-detection
        combined_exclusions = exclusion_mask | z_boundary_mask
        
        if np.any(combined_exclusions):
            excluded_voxels = np.sum(combined_exclusions)
            total_threshold_voxels = np.sum(coarse_mask)
            coarse_mask = coarse_mask & ~combined_exclusions
            remaining_voxels = np.sum(coarse_mask)
            print(f"Excluded {excluded_voxels:,} voxels from {total_threshold_voxels:,} threshold voxels → {remaining_voxels:,} remain")
        else:
            print(f"No pre-detection exclusions applied")

        # 2) Initial CCL
        struct = connectivity_structure(connectivity)
        initial_labels, n_init = ndi.label(coarse_mask, structure=struct)
        print(f"Initial threshold found {n_init} connected blobs")

        # 2.5) PRE-FILTER: drop tiny initial blobs (speckle cleanup)
        if pre_min_voxels is not None:
            initial_labels, removed_pre = filter_small_components(
                initial_labels, min_voxels=pre_min_voxels
            )
            n_after_pre = len(np.unique(initial_labels)) - 1  # Subtract 1 for background (label 0)
            print(f"Pre-filter removed {removed_pre} tiny blobs → {n_after_pre} blobs remain")
        else:
            removed_pre = 0

        # 3) Split each remaining component via watershed
        split_labelmap = np.zeros_like(initial_labels, dtype=np.int32)
        next_global = 1
        crumbs_removed = 0

        for comp_id in range(1, int(initial_labels.max()) + 1):
            comp_mask = (initial_labels == comp_id)
            if not comp_mask.any():
                continue
            zsl, ysl, xsl = bbox_from_mask(comp_mask)
            comp_sub = comp_mask[zsl, ysl, xsl]
            suv_sub  = self.suv_data[zsl, ysl, xsl]

            ws_sub = split_component_watershed(
                suv_sub=suv_sub,
                comp_mask_sub=comp_sub,
                component_id=comp_id)

            # 3.5) POST-FILTER: drop tiny watershed pieces before global paste
            _post_min_voxels = post_min_voxels if post_min_voxels is not None else min_voxels

            if ws_sub.max() > 0:
                # fast per-label counts
                labs = ws_sub[ws_sub > 0]
                counts = np.bincount(labs, minlength=int(ws_sub.max())+1)
                for k in range(1, int(ws_sub.max())+1):
                    if counts[k] == 0:
                        continue
                    vol_vox = int(counts[k])
                    # Drop if too small in voxels
                    if vol_vox < _post_min_voxels:
                        ws_sub[ws_sub == k] = 0
                        crumbs_removed += 1

            # paste kept subregions
            for k in np.unique(ws_sub):
                if k == 0:
                    continue
                sub = split_labelmap[zsl, ysl, xsl]
                sub[ws_sub == k] = next_global
                split_labelmap[zsl, ysl, xsl] = sub
                next_global += 1

        # 4) Final CCL (guarantee contiguity & tidy IDs)
        final_labels = final_ccl_from_labelmap(split_labelmap, connectivity=connectivity)
        unique_final = np.unique(final_labels)
        unique_final = unique_final[unique_final > 0]
        print(f"Final CCL produced {len(unique_final)} regions")

        # 4.5) Compute liver SUVmean for lesion filtering
        self.liver_suv_stats = compute_liver_suv(self.organ_data, self.suv_data, liver_label=5)
        liver_suv = self.liver_suv_stats['liver_suv_threshold']

        # 5) Build Lesion objects; apply final size filter and liver SUVmean filter
        lesions = []
        size_filtered_count = 0
        liver_filtered_count = 0

        for lesion_id in unique_final:
            lesion_voxels = (final_labels == lesion_id)
            volume_voxels = int(lesion_voxels.sum())

            # Check for size thresholds (voxels only)
            if volume_voxels < min_voxels:
                size_filtered_count += 1
                # Remove from final_labels immediately
                final_labels[lesion_voxels] = 0
                continue
            
            # Check liver SUV threshold
            max_suv = float(np.max(self.suv_data[lesion_voxels]))
            if max_suv < liver_suv: 
                liver_filtered_count += 1
                # Remove from final_labels immediately
                final_labels[lesion_voxels] = 0
                continue

            coords = np.where(lesion_voxels)
            # np.where returns (dim0_indices, dim1_indices, dim2_indices)
            # For medical imaging: dim0=x, dim1=y, dim2=z
            # Calculate center as [x, y, z] format for consistency
            center_coords = np.array([np.mean(coords[0]), np.mean(coords[1]), np.mean(coords[2])])
            # max_suv already calculated above for liver filtering
            mean_suv = float(np.mean(self.suv_data[lesion_voxels]))

            min_coords = [int(np.min(c)) for c in coords]
            max_coords = [int(np.max(c)) for c in coords]
            bounding_box = tuple(slice(min_coords[i], max_coords[i] + 1) for i in range(3))

            organ_overlaps = analyze_organ_overlaps(self.organ_data, self.organ_mapping, lesion_voxels)
            self._ensure_nearest_fields()
            closest_organs = find_closest_organs(lesion_voxels, self.organ_data, self.organ_mapping,
                                               self.nearest_organ_dist_voxels, self.nearest_organ_label,
                                               self.suv_data.shape, num_closest=5)
            combined_organs = combine_organ_results(organ_overlaps, closest_organs, target_count=3)
            vertebrae_level = determine_vertebrae_level(lesion_voxels, self.vertebrae_axial_ranges) if hasattr(self, 'vertebrae_axial_ranges') else None

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

        print(f"Filtered out {size_filtered_count} small regions by size")
        print(f"Filtered out {liver_filtered_count} regions with SUVmax < liver SUV ({liver_suv:.2f})")
        print(f"Extracted {len(lesions)} clinically significant lesions")
        
        # Create clean final lesion mask with only kept lesions
        self.final_lesion_mask = np.zeros_like(final_labels, dtype=np.int32)
        for i, lesion in enumerate(lesions, 1):
            lesion_voxels = (final_labels == lesion.id)
            self.final_lesion_mask[lesion_voxels] = i  # Relabel sequentially
        
        # Verify the mask has the correct number of lesions
        unique_mask_labels = np.unique(self.final_lesion_mask)
        unique_mask_labels = unique_mask_labels[unique_mask_labels > 0]
        print(f"Final mask contains {len(unique_mask_labels)} lesion regions")
        
        
        return lesions

    def save_lesion_mask(self, output_path: str) -> None:
        """
        Save the final lesion segmentation mask as a NIfTI file
        
        Args:
            output_path: Path where to save the lesion mask (should end with .nii or .nii.gz)
        """
        if not hasattr(self, 'final_lesion_mask'):
            raise ValueError("No lesion mask available. Run extract_lesions() first.")
        
        save_lesion_mask(self.final_lesion_mask, output_path, self.suv_img.affine, self.suv_img.header)
    
    def run_pipeline(self, suv_threshold: float = 2.5, grow_suv_threshold: float = 2.5,
                    min_voxels: int = 20, connectivity: int = 18,
                    pre_min_voxels: int = 20, post_min_voxels: Optional[int] = None,
                    save_mask_path: Optional[str] = None, exclude_brain: bool = False, 
                    exclude_kidneys: bool = False, exclude_bladder: bool = False, 
                    z_boundary_voxels: int = 2) -> List[Lesion]:
        """
        Run the complete pipeline
        
        Args:
            suv_threshold: Minimum SUV value for lesion detection
            min_voxels: Minimum number of voxels for lesion detection
            save_mask_path: Optional path to save the final lesion segmentation mask
            exclude_brain: Whether to exclude brain from lesion detection
            exclude_kidneys: Whether to exclude kidneys from lesion detection
            exclude_bladder: Whether to exclude bladder from lesion detection
            z_boundary_voxels: Number of voxels to exclude from Z-axis boundaries
            
        Returns:
            List of extracted lesions
        """
        print("Starting PET lesion extraction pipeline...")
        print(f"Filtering parameters:")
        print(f"  SUV threshold: {suv_threshold}")
        print(f"  Minimum voxels: {min_voxels}")
        
        # Extract lesions with timing
        import time
        start_time = time.time()
        lesions = self.extract_lesions(
                                       suv_threshold=suv_threshold,
                                       grow_suv_threshold=grow_suv_threshold,
                                       min_voxels=min_voxels,
                                       connectivity=connectivity,
                                       pre_min_voxels=pre_min_voxels,
                                       post_min_voxels=post_min_voxels,
                                       exclude_brain=exclude_brain, exclude_kidneys=exclude_kidneys, 
                                       exclude_bladder=exclude_bladder, z_boundary_voxels=z_boundary_voxels)
        processing_time = time.time() - start_time
        
        print(f"Pipeline completed in {processing_time:.2f} seconds")
        
        # Save lesion mask if requested
        if save_mask_path:
            self.save_lesion_mask(save_mask_path)
        
        return lesions
        return lesions

    def _expand_labels(self, labels: np.ndarray, suv_threshold: float = 2.5, max_distance: int = 2) -> np.ndarray:
        """
        Expand lesion labels to include immediate neighbors with SUV > threshold.
        
        Args:
            labels: Current label map
            suv_threshold: SUV threshold for inclusion
            max_distance: Maximum expansion distance in voxels
            
        Returns:
            Expanded label map
        """
        current_labels = labels.copy()
        struct = ndi.generate_binary_structure(3, 3)  # 26-connectivity
        
        for i in range(max_distance):
            # Dilate current mask to find neighbors
            dilated_labels = ndi.maximum_filter(current_labels, footprint=struct)
            
            # Identify candidates: background voxels, neighbor to a label, and SUV >= threshold
            candidates_mask = (current_labels == 0) & (dilated_labels > 0) & (self.suv_data >= suv_threshold)
            
            num_new_voxels = np.sum(candidates_mask)
            
            if num_new_voxels == 0:
                break
                
            # Assign the label of the neighbor (using the dilated label map which has the max neighbor label)
            current_labels[candidates_mask] = dilated_labels[candidates_mask]
            
        return current_labels
def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="PET Lesion Extraction Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode selection
    parser.add_argument("--input_mask_file", default=None,
                       help="Path to existing lesion segmentation mask (if provided, will analyze this instead of extracting)")
    
    # Input files
    parser.add_argument("--suv_file", "-s", required=True,
                       help="Path to SUV NIfTI file")
    parser.add_argument("--organ_mask_dir", "-o",
                       required=True,
                       help="Directory containing organ mask NIfTI files")
    parser.add_argument("--organ_mask_prefix", 
                       required=True,
                       help="Prefix for organ mask files (will append mask type names)")
    parser.add_argument("--mapping_file", "-m",
                       default="data/totalsegmentator_index_mapping.json",
                       help="Path to organ mapping JSON file")
    
    # Pipeline parameters
    parser.add_argument("--suv_threshold", "-t", type=float, default=2.5,
                       help="Minimum SUV value for lesion detection")
    # Removed min_volume_mm3 argument - now using voxels only
    parser.add_argument("--min_voxels", "-p", type=int, default=20,
                       help="Minimum number of voxels for lesion detection")
    
    # Output options
    parser.add_argument("--output_json", "-j", default="lesion_results.json",
                       help="Output JSON file for lesion results")
    parser.add_argument("--save_mask", "-k", default="lesion_mask.nii",
                       help="Path to save lesion segmentation mask (optional)")
    
    # Advanced options
    parser.add_argument("--connectivity", type=int, default=18, choices=[6, 18, 26],
                       help="Connectivity for lesion segmentation (6, 18, or 26)")
    parser.add_argument("--pre_min_voxels", type=int, default=20,
                       help="Pre-filter minimum voxels for initial blob cleanup")
    parser.add_argument("--post_min_voxels", type=int, default=None,
                       help="Post-filter minimum voxels for watershed pieces")
    
    # Exclusion options
    parser.add_argument("--exclude_brain", type=lambda x: x.lower() in ['true', '1', 'yes'], 
                       default=False, help="Exclude brain from lesion detection (default: False)")
    parser.add_argument("--exclude_kidneys", type=lambda x: x.lower() in ['true', '1', 'yes'], 
                       default=False, help="Exclude kidneys from lesion detection (default: False)")
    parser.add_argument("--exclude_bladder", type=lambda x: x.lower() in ['true', '1', 'yes'], 
                       default=False, help="Exclude bladder from lesion detection (default: False)")
    parser.add_argument("--no_exclusions", action="store_true", default=False,
                       help="Disable all organ exclusions (overrides individual settings)")
    parser.add_argument("--z_boundary_voxels", type=int, default=2,
                       help="Number of voxels to exclude from Z-axis boundaries (default: 2)")
    
    return parser.parse_args()

def run_suv_extraction(suv_file: str, organ_mask_files: dict, mapping_file: str,
                      output_json: str, save_mask_path: str = None,
                      suv_threshold: float = 2.5, grow_suv_threshold: float = 2.5,
                      min_voxels: int = 20,
                      connectivity: int = 18, pre_min_voxels: int = 20,
                      post_min_voxels: int = None, exclude_brain: bool = False,
                      exclude_kidneys: bool = False, exclude_bladder: bool = False,
                      z_boundary_voxels: int = 2) -> tuple:
    """
    Core SUV extraction function without CLI parsing
    
    Args:
        suv_file: Path to SUV NIfTI file
        organ_mask_files: Dictionary of mask type to file path
        mapping_file: Path to organ mapping JSON file
        output_json: Path to save lesion results JSON
        save_mask_path: Optional path to save lesion mask
        **kwargs: Other parameters for lesion extraction
        
    Returns:
        Tuple of (lesions, output_data) where output_data is the complete JSON structure
    """
    print(f"Running PET Lesion Extraction Pipeline")
    print(f"Input files:")
    print(f"  SUV file: {suv_file}")
    print(f"  Organ mask files:")
    for mask_type, mask_file in organ_mask_files.items():
        print(f"    {mask_type}: {mask_file}")
    print(f"  Mapping file: {mapping_file}")
    print(f"Parameters:")
    print(f"  SUV threshold: {suv_threshold}")
    print(f"  Min voxels: {min_voxels}")
    if save_mask_path:
        print(f"  Mask output: {save_mask_path}")
    print(f"  JSON output: {output_json}")
    
    # Handle exclusion settings
    excluded_organs = []
    if exclude_brain: excluded_organs.append("brain")
    if exclude_kidneys: excluded_organs.append("kidneys")
    if exclude_bladder: excluded_organs.append("bladder")
    print(f"  Excluding organs: {', '.join(excluded_organs) if excluded_organs else 'none'}")
    
    print()
    
    # Initialize and run pipeline
    pipeline = PETLesionPipeline(suv_file, organ_mask_files, mapping_file)
    
    lesions = pipeline.run_pipeline(
        suv_threshold=suv_threshold,
        grow_suv_threshold=grow_suv_threshold,
        min_voxels=min_voxels,
        connectivity=connectivity,
        pre_min_voxels=pre_min_voxels,
        post_min_voxels=post_min_voxels,
        save_mask_path=save_mask_path,
        exclude_brain=exclude_brain,
        exclude_kidneys=exclude_kidneys,
        exclude_bladder=exclude_bladder,
        z_boundary_voxels=z_boundary_voxels
    )
    
    # Save results to JSON with metadata
    lesion_data = []
    
    # Collect pipeline metadata
    pipeline_metadata = {
        'pipeline_version': '1.0',
        'processing_parameters': {
            'suv_threshold': suv_threshold,
            'grow_suv_threshold': grow_suv_threshold,
            'min_voxels': min_voxels,
            'connectivity': connectivity,
            'z_boundary_voxels': z_boundary_voxels,
            'exclude_brain': exclude_brain,
            'exclude_kidneys': exclude_kidneys,
            'exclude_bladder': exclude_bladder
        },
        'image_info': {
            'suv_file': suv_file,
            'image_shape': list(pipeline.suv_data.shape),
            'voxel_size_mm': list(pipeline.voxel_size),
            'total_voxels': int(np.prod(pipeline.suv_data.shape))
        },
        'organ_masks': {
            'mask_files': organ_mask_files,
            'mapping_file': mapping_file,
            'total_organs_loaded': len(pipeline.organ_mapping),
            'mask_types': list(organ_mask_files.keys())
        },
        'liver_suv_analysis': getattr(pipeline, 'liver_suv_stats', {
            'liver_suv_threshold': 'N/A',
            'liver_filtering_applied': False
        }),
        'lesion_statistics': {
            'total_lesions_found': len(lesions),
            'coordinate_system': {
                'description': 'Coordinates centered at image center',
                'image_center_voxels': [
                    pipeline.suv_data.shape[0] / 2.0,
                    pipeline.suv_data.shape[1] / 2.0, 
                    pipeline.suv_data.shape[2] / 2.0
                ],
                'format': '[x, y, z] where (0,0,0) is image center'
            }
        }
    }
    
    # Get image center for coordinate system centering
    # lesion.coordinates is [x, y, z] in voxel indices
    # We want x, y, z coordinates centered at image center
    image_center_x = pipeline.suv_data.shape[0] / 2.0  # Center of x-axis (dim0)
    image_center_y = pipeline.suv_data.shape[1] / 2.0  # Center of y-axis (dim1)
    image_center_z = pipeline.suv_data.shape[2] / 2.0  # Center of z-axis (dim2)
    
    for i, lesion in enumerate(lesions, 1):  # Renumber from 1 to N
        # Convert lesion coordinates to centered system
        # lesion.coordinates = [x, y, z], so we can directly apply centering
        centered_x = float(lesion.coordinates[0] - image_center_x)  # x-coord centered
        centered_y = float(lesion.coordinates[1] - image_center_y)  # y-coord centered
        centered_z = float(lesion.coordinates[2] - image_center_z)  # z-coord centered
        
        lesion_dict = {
            'id': i,  # Sequential ID from 1 to len(lesions)
            'center_coords': [centered_x, centered_y, centered_z],  # [x, y, z] centered at image center
            'volume_voxels': int(lesion.volume_voxels),
            'max_suv': float(lesion.max_suv),
            'mean_suv': float(lesion.mean_suv),
            'organ_overlaps': {k: f"{float(v):.2f}%" for k, v in lesion.organ_overlaps.items()},
            'closest_organs': {k: {
                'distance_voxels': float(v['distance_voxels']),
                'spatial_relationship': v.get('spatial_relationship', {})
            } for k, v in lesion.closest_organs.items()},
            'vertebrae_level': lesion.vertebrae_level if lesion.vertebrae_level else []
        }
        
        # Add shape metrics
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
    
    # Create final output structure with metadata and lesions
    output_data = {
        'metadata': pipeline_metadata,
        'lesions': lesion_data
    }
    
    # Convert NumPy types to Python native types for JSON serialization
    output_data_clean = convert_numpy_types(output_data)
    
    with open(output_json, 'w') as f:
        json.dump(output_data_clean, f, indent=2)
    
    print(f"Saved {len(lesion_data)} lesions and metadata to JSON file: {output_json}")
    
    return lesions, output_data_clean


def analyze_existing_segmentation(
    input_mask_file: str,
    suv_file: str,
    organ_mask_files: dict,
    mapping_file: str,
    output_json: str
) -> Dict[str, Any]:
    """
    Analyze an existing lesion segmentation mask and produce JSON output.
    
    Args:
        input_mask_file: Path to existing lesion segmentation mask NIfTI file
        suv_file: Path to SUV NIfTI file
        organ_mask_files: Dictionary of mask type to file path
        mapping_file: Path to organ mapping JSON file
        output_json: Path to save lesion results JSON
        
    Returns:
        Dictionary containing the output data
    """
    print(f"Analyzing existing lesion segmentation mask")
    print(f"Input files:")
    print(f"  Lesion mask: {input_mask_file}")
    print(f"  SUV file: {suv_file}")
    print(f"  Organ mask files:")
    for mask_type, mask_file in organ_mask_files.items():
        print(f"    {mask_type}: {mask_file}")
    print(f"  Mapping file: {mapping_file}")
    print()
    
    # Load the existing lesion mask
    print("Loading lesion segmentation mask...")
    lesion_mask_img = nib.load(input_mask_file)
    lesion_mask = lesion_mask_img.get_fdata().astype(np.int32)
    
    # Initialize pipeline to get organ data and SUV data
    print("Initializing pipeline for organ and SUV analysis...")
    pipeline = PETLesionPipeline(suv_file, organ_mask_files, mapping_file)
    
    # Get unique lesion labels
    unique_labels = np.unique(lesion_mask)
    unique_labels = unique_labels[unique_labels > 0]
    print(f"Found {len(unique_labels)} lesions in the mask")
    
    # Analyze each lesion
    lesions = []
    for lesion_id in unique_labels:
        lesion_voxels = (lesion_mask == lesion_id)
        volume_voxels = int(lesion_voxels.sum())
        
        # Get coordinates
        coords = np.where(lesion_voxels)
        center_coords = np.array([np.mean(coords[0]), np.mean(coords[1]), np.mean(coords[2])])
        
        # Get SUV values
        max_suv = float(np.max(pipeline.suv_data[lesion_voxels]))
        mean_suv = float(np.mean(pipeline.suv_data[lesion_voxels]))
        
        # Get bounding box
        min_coords = [int(np.min(c)) for c in coords]
        max_coords = [int(np.max(c)) for c in coords]
        bounding_box = tuple(slice(min_coords[i], max_coords[i] + 1) for i in range(3))
        
        # Analyze organ overlaps
        organ_overlaps = analyze_organ_overlaps(pipeline.organ_data, pipeline.organ_mapping, lesion_voxels)
        
        # Find closest organs
        pipeline._ensure_nearest_fields()
        closest_organs = find_closest_organs(
            lesion_voxels, pipeline.organ_data, pipeline.organ_mapping,
            pipeline.nearest_organ_dist_voxels, pipeline.nearest_organ_label,
            pipeline.suv_data.shape, num_closest=5
        )
        
        # Combine organ results
        combined_organs = combine_organ_results(organ_overlaps, closest_organs, target_count=3)
        
        # Determine vertebrae level
        vertebrae_level = determine_vertebrae_level(lesion_voxels, pipeline.vertebrae_axial_ranges) if hasattr(pipeline, 'vertebrae_axial_ranges') else None
        
        # Compute shape metrics
        shape = compute_shape_metrics(lesion_voxels.astype(bool))
        
        # Create lesion object
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
    
    print(f"Analyzed {len(lesions)} lesions")
    
    # Build output data structure
    pipeline_metadata = {
        'pipeline_version': '1.0',
        'analysis_mode': 'existing_segmentation',
        'processing_parameters': {
            'input_mask_file': input_mask_file,
            'analysis_type': 'post-hoc'
        },
        'image_info': {
            'suv_file': suv_file,
            'image_shape': list(pipeline.suv_data.shape),
            'voxel_size_mm': list(pipeline.voxel_size),
            'total_voxels': int(np.prod(pipeline.suv_data.shape))
        },
        'organ_masks': {
            'mask_files': organ_mask_files,
            'mapping_file': mapping_file,
            'total_organs_loaded': len(pipeline.organ_mapping),
            'mask_types': list(organ_mask_files.keys())
        },
        'liver_suv_analysis': getattr(pipeline, 'liver_suv_stats', {
            'liver_suv_threshold': 'N/A',
            'liver_filtering_applied': False
        }),
        'lesion_statistics': {
            'total_lesions_found': len(lesions),
            'coordinate_system': {
                'description': 'Coordinates centered at image center',
                'image_center_voxels': [
                    pipeline.suv_data.shape[0] / 2.0,
                    pipeline.suv_data.shape[1] / 2.0, 
                    pipeline.suv_data.shape[2] / 2.0
                ],
                'format': '[x, y, z] where (0,0,0) is image center'
            }
        }
    }
    
    # Get image center for coordinate system centering
    image_center_x = pipeline.suv_data.shape[0] / 2.0
    image_center_y = pipeline.suv_data.shape[1] / 2.0
    image_center_z = pipeline.suv_data.shape[2] / 2.0
    
    lesion_data = []
    for i, lesion in enumerate(lesions, 1):
        # Convert lesion coordinates to centered system
        centered_x = float(lesion.coordinates[0] - image_center_x)
        centered_y = float(lesion.coordinates[1] - image_center_y)
        centered_z = float(lesion.coordinates[2] - image_center_z)
        
        lesion_dict = {
            'id': i,
            'center_coords': [centered_x, centered_y, centered_z],
            'volume_voxels': int(lesion.volume_voxels),
            'max_suv': float(lesion.max_suv),
            'mean_suv': float(lesion.mean_suv),
            'organ_overlaps': {k: f"{float(v):.2f}%" for k, v in lesion.organ_overlaps.items()},
            'closest_organs': {k: {
                'distance_voxels': float(v['distance_voxels']),
                'spatial_relationship': v.get('spatial_relationship', {})
            } for k, v in lesion.closest_organs.items()},
            'vertebrae_level': lesion.vertebrae_level if lesion.vertebrae_level else []
        }
        
        # Add shape metrics
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
    
    # Create final output structure
    output_data = {
        'metadata': pipeline_metadata,
        'lesions': lesion_data
    }
    
    # Convert NumPy types to Python native types for JSON serialization
    output_data_clean = convert_numpy_types(output_data)
    
    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(output_data_clean, f, indent=2)
    
    print(f"Saved {len(lesion_data)} lesions to JSON file: {output_json}")
    
    return output_data_clean


def main():
    """Main function to run the pipeline"""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Check if mapping file exists and load it to get mask types
    if not os.path.exists(args.mapping_file):
        print(f"Error: Mapping file not found: {args.mapping_file}")
        return
    
    # Load mapping file to get available mask types
    with open(args.mapping_file, 'r') as f:
        mapping_data = json.load(f)
    
    # Construct organ mask file paths
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
    
    # Check if SUV file exists
    if not os.path.exists(args.suv_file):
        print(f"Error: SUV file not found: {args.suv_file}")
        return
    
    # Check if we're in analysis mode (existing mask) or extraction mode
    if args.input_mask_file:
        # Analysis mode: analyze existing segmentation mask
        print("=" * 80)
        print("MODE: ANALYZING EXISTING SEGMENTATION MASK")
        print("=" * 80)
        
        if not os.path.exists(args.input_mask_file):
            print(f"Error: Input mask file not found: {args.input_mask_file}")
            return
        
        output_data = analyze_existing_segmentation(
            input_mask_file=args.input_mask_file,
            suv_file=args.suv_file,
            organ_mask_files=organ_mask_files,
            mapping_file=args.mapping_file,
            output_json=args.output_json
        )
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Output JSON: {args.output_json}")
        
    else:
        # Extraction mode: run full pipeline
        print("=" * 80)
        print("MODE: LESION EXTRACTION PIPELINE")
        print("=" * 80)
        
        # Handle exclusion settings
        if args.no_exclusions:
            exclude_brain = exclude_kidneys = exclude_bladder = False
        else:
            exclude_brain = args.exclude_brain
            exclude_kidneys = args.exclude_kidneys  
            exclude_bladder = args.exclude_bladder
        
        # Call the core function
        lesions, output_data = run_suv_extraction(
            suv_file=args.suv_file,
            organ_mask_files=organ_mask_files,
            mapping_file=args.mapping_file,
            output_json=args.output_json,
            save_mask_path=args.save_mask,
            suv_threshold=args.suv_threshold,
            min_voxels=args.min_voxels,
            connectivity=args.connectivity,
            pre_min_voxels=args.pre_min_voxels,
            post_min_voxels=args.post_min_voxels,
            exclude_brain=exclude_brain,
            exclude_kidneys=exclude_kidneys,
            exclude_bladder=exclude_bladder,
            z_boundary_voxels=args.z_boundary_voxels
        )

if __name__ == "__main__":
    main()
