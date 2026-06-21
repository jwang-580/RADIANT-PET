#!/usr/bin/env python3
"""
Generate lesion description JSON from a lesion mask and SUV image.
Uses utils.candidate_analysis.MaskAnalyzer to perform the analysis.
Supports batch processing of multiple cases.
"""

import os
import sys
import json
import argparse
import numpy as np
import glob
from pathlib import Path

# Add project root and utils directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'utils'))

from utils.candidate_analysis import MaskAnalyzer
from utils.lesion_features import convert_numpy_types
from suv_threshold_processing.rule_based_postprocessing import apply_rule2_symmetric_detection

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate lesion description from existing mask(s)")
    
    parser.add_argument("--lesion_mask", "-l", default="nnunet/nnUNet_raw/Dataset001_autopet/labelstr/nnunet_t8/", help="Path to lesion segmentation NIfTI file or directory (if batch)")
    parser.add_argument("--organ_mask_dir", "-o", default="nnunet/nnUNet_raw/Dataset001_autopet/imagesTr/", help="Directory containing organ mask NIfTI files")
    parser.add_argument("--organ_mask_prefix", default=None, help="Prefix for organ mask files. If None in batch mode, uses input filename stem + '_'")
    parser.add_argument("--mapping_file", "-m", default="data/totalsegmentator_index_mapping.json", help="Path to organ mapping JSON file")
    parser.add_argument("--suv_file", "-s", default="nnunet/nnUNet_raw/Dataset001_autopet/imagesTr/", help="Optional path to SUV NIfTI file or directory (if batch) for intensity stats")
    parser.add_argument("--output_json", "-j", default="nnunet/nnUNet_raw/Dataset001_autopet/results/", help="Output JSON file or directory (if batch)")
    
    # Symmetry parameters
    parser.add_argument("--y_threshold", type=float, default=2.0, help="Y coordinate difference threshold for symmetry (default: 2.0)")
    parser.add_argument("--z_threshold", type=float, default=1.0, help="Z coordinate difference threshold for symmetry (default: 1.0)")
    parser.add_argument("--x_sum_threshold", type=float, default=3.0, help="X coordinate sum threshold for bilateral symmetry (default: 3.0)")
    parser.add_argument("--suv_threshold", type=float, default=1.0, help="SUV difference threshold for symmetry (default: 1.0)")
    parser.add_argument("--shape_threshold", type=float, default=0.3, help="Shape similarity threshold for symmetry (default: 0.3)")

    parser.add_argument("--batch", action="store_true", help="Enable batch processing mode")
    
    return parser.parse_args()

def process_single_case(
    lesion_mask_path,
    suv_file_path,
    output_json_path,
    args,
    current_organ_prefix=None,
    organ_mask_files_override=None,
):
    """
    Process a single case.
    """
    print(f"Processing: {lesion_mask_path}")
    
    # Determine organ mask prefix
    organ_prefix = args.organ_mask_prefix
    if current_organ_prefix:
        organ_prefix = current_organ_prefix
    elif organ_prefix is None:
        raise ValueError("An organ-mask prefix is required in single-case mode")

    # Load mapping to get mask types
    with open(args.mapping_file, 'r') as f:
        mapping_data = json.load(f)
        
    organ_mask_files = dict(organ_mask_files_override or {})
    if not organ_mask_files:
        for mask_type in mapping_data.keys():
            mask_file_path = os.path.join(args.organ_mask_dir, f"{organ_prefix}0000_{mask_type}.nii")
            if not os.path.exists(mask_file_path):
                 # Try with .nii.gz
                 mask_file_path = os.path.join(args.organ_mask_dir, f"{organ_prefix}0000_{mask_type}.nii.gz")

            if os.path.exists(mask_file_path):
                organ_mask_files[mask_type] = mask_file_path
            
    if not organ_mask_files:
        print(f"Error: No organ mask files found for prefix '{organ_prefix}' in {args.organ_mask_dir}")
        return False

    # Run Analysis
    try:
        analyzer = MaskAnalyzer(
            lesion_mask_file=lesion_mask_path,
            organ_mask_files=organ_mask_files,
            mapping_file=args.mapping_file,
            suv_file=suv_file_path
        )
        
        lesions = analyzer.analyze_lesions()
        
        # Format output to match the threshold pipeline.
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
        
        # Apply symmetry detection
        print("Running symmetry detection...")
        lesion_data, symmetric_count = apply_rule2_symmetric_detection(
            lesion_data,
            y_threshold=args.y_threshold,
            z_threshold=args.z_threshold,
            x_sum_threshold=args.x_sum_threshold,
            suv_threshold=args.suv_threshold,
            shape_threshold=args.shape_threshold
        )
        print(f"Found {symmetric_count} symmetric lesion pairs")

        output_data = {
            'metadata': {
                'lesion_mask': lesion_mask_path,
                'suv_file': suv_file_path,
                'total_lesions': len(lesions),
                'image_shape': list(analyzer.shape),
                'voxel_size': list(analyzer.voxel_size),
                'organ_masks': {
                    'mask_files': organ_mask_files,
                    'mapping_file': args.mapping_file,
                    'total_organs_loaded': len(analyzer.organ_mapping)
                },
                'lesion_statistics': {
                    'total_lesions_found': len(lesions),
                    'coordinate_system': {
                        'description': 'Coordinates centered at image center',
                        'image_center_voxels': [image_center_x, image_center_y, image_center_z],
                        'format': '[x, y, z] where (0,0,0) is image center'
                    },
                    'symmetry_parameters': {
                        'y_threshold': args.y_threshold,
                        'z_threshold': args.z_threshold,
                        'x_sum_threshold': args.x_sum_threshold,
                        'suv_threshold': args.suv_threshold,
                        'shape_threshold': args.shape_threshold
                    }
                }
            },
            'lesions': lesion_data
        }
        
        # Convert NumPy types
        output_data = convert_numpy_types(output_data)
        
        # Save output
        output_dir = os.path.dirname(output_json_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"Saved analysis to {output_json_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {lesion_mask_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def find_suv_file(case_id, suv_input):
    """
    Find corresponding SUV file for a case ID.
    suv_input can be a file (not useful in match) or directory.
    """
    if not suv_input:
        return None
        
    if os.path.isfile(suv_input):
        return suv_input
        
    if os.path.isdir(suv_input):
        # Look for files starting with case_id
        # Common patterns: {case_id}.nii.gz, {case_id}_0001.nii.gz (nnUNet convention)
        potential_files = [
            os.path.join(suv_input, f"{case_id}_0001.nii.gz"),
            os.path.join(suv_input, f"{case_id}_400_0001.nii.gz")
        ]
        
        for f in potential_files:
            if os.path.exists(f):
                return f
                
        return None
        
    return None

def main():
    args = parse_arguments()
    
    # Handle Batch Mode
    if args.batch:
        if not os.path.isdir(args.lesion_mask):
            print(f"Error: In batch mode, --lesion_mask must be a directory. Got: {args.lesion_mask}")
            sys.exit(1)
            
        if not os.path.exists(args.output_json):
            os.makedirs(args.output_json)
        elif not os.path.isdir(args.output_json):
             print(f"Error: In batch mode, --output_json must be a directory (or not exist). Got file: {args.output_json}")
             sys.exit(1)
             
        # Iterate over files
        lesion_files = sorted(glob.glob(os.path.join(args.lesion_mask, "*_watershed.nii.gz"))) + \
                       sorted(glob.glob(os.path.join(args.lesion_mask, "*_watershed.nii")))
                       
        print(f"Found {len(lesion_files)} lesion masks in {args.lesion_mask}")
        
        success_count = 0
        
        for lesion_file in lesion_files:
            filename = os.path.basename(lesion_file)
            
            # Identify Case ID (stem)
            # specified format: OSU?_acc?
            # Example: OSU02_acc0_CT_400.nii.gz -> OSU02_acc0
            # Example: OSU02_acc0.nii.gz -> OSU02_acc0
            
            # Remove extensions
            base_name = filename
            if base_name.endswith(".nii.gz"):
                base_name = base_name[:-7]
            elif base_name.endswith(".nii"):
                base_name = base_name[:-4]
                
            # split by underscore
            parts = base_name.split('_')
            
            # Reconstruct OSU part and acc part
            # Look for parts starting with "OSU" and "acc"
            osu_part = None
            acc_part = None
            
            for part in parts:
                if part.upper().startswith("OSU"):
                    osu_part = part
                elif part.startswith("acc"):
                    acc_part = part
            
            if osu_part and acc_part:
                case_id = f"{osu_part}_{acc_part}"
            else:
                # If no "osu" in name, remove mask-related suffixes
                case_id = base_name
                for suffix in ['_t8_mask', '_t8_mask_watershed', '_pred', '_prediction', '_filtered']:
                    if case_id.endswith(suffix):
                        case_id = case_id[:-len(suffix)]
                        break
      
            # output file
            output_json_path = os.path.join(args.output_json, f"{case_id}_description.json")
            
            # Find SUV file
            suv_file_path = find_suv_file(case_id, args.suv_file)
            
            # Determine organ prefix
            # If didn't specify prefix, assume {case_id}_
            current_prefix = args.organ_mask_prefix
            if current_prefix is None:
                current_prefix = f"{case_id}_"
                
            print(f"\n--- Case: {case_id} ---")
            if process_single_case(lesion_file, suv_file_path, output_json_path, args, current_prefix):
                success_count += 1
                
        print(f"\nBatch processing complete. Successfully processed {success_count}/{len(lesion_files)} files.")
    
    else:
        # Single file mode
        if not os.path.exists(args.lesion_mask):
            print(f"Error: Lesion mask not found: {args.lesion_mask}")
            sys.exit(1)
            
        if not os.path.exists(args.mapping_file):
            print(f"Error: Mapping file not found: {args.mapping_file}")
            sys.exit(1)
            
        if args.organ_mask_prefix is None:
            print("Error: --organ_mask_prefix is required in single-case mode")
            sys.exit(1)
        prefix = args.organ_mask_prefix
            
        process_single_case(args.lesion_mask, args.suv_file, args.output_json, args, prefix)

if __name__ == "__main__":
    main()
