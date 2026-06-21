#!/usr/bin/env python3
"""
Batch processing script for combined PET-CT lesion analysis pipeline.

Processes all SUV files in a directory using the combined pipeline with
consistent naming patterns and default settings.
"""

import os
import sys
import glob
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from suv_threshold_processing.threshold_pipeline import run_combined_pipeline
import json


def extract_patient_id(suv_file: str) -> str:
    """
    Extract patient ID from SUV filename.
    
    Args:
        suv_file: Path to SUV file (e.g., /path/to/abc123_0001.nii.gz)
        
    Returns:
        Patient ID (e.g., abc123)
    """
    basename = os.path.basename(suv_file)
    # Remove _0001.nii.gz to get patient ID
    patient_id = basename.replace('_0001.nii.gz', '')
    return patient_id


def process_single_patient(
    patient_id: str,
    suv_dir: str,
    organ_mask_dir: str,
    output_dir: str,
    mapping_file: str,
    exclude_brain: bool = True,
    exclude_kidneys: bool = False,
    exclude_bladder: bool = True,
    input_mask_dir: str = None,
    input_mask_suffix: str = None,
    **kwargs
) -> bool:
    """
    Process a single patient through the combined pipeline.
    
    Args:
        patient_id: Patient identifier
        suv_dir: Directory containing SUV files
        organ_mask_dir: Directory containing organ mask files
        output_dir: Directory for output files
        mapping_file: Path to organ mapping JSON file
        exclude_brain: Whether to exclude brain (default: True)
        exclude_kidneys: Whether to exclude kidneys (default: False)
        exclude_bladder: Whether to exclude bladder (default: True)
        **kwargs: Additional parameters to pass to pipeline
        
    Returns:
        True if successful, False otherwise
    """
    # Construct file paths
    suv_file = os.path.join(suv_dir, f"{patient_id}_0001.nii.gz")
    gt_mask_file = os.path.join(suv_dir, f"{patient_id}_gt.nii.gz")
    output_json = os.path.join(output_dir, f"{patient_id}_results.json")
    output_mask = os.path.join(output_dir, f"{patient_id}_mask.nii")
    
    # Determine input mask file (if in analysis mode)
    input_mask_file = None
    if input_mask_dir:
        # If suffix not provided, try to infer or use default
        suffix = input_mask_suffix if input_mask_suffix else "_lesion_mask.nii.gz"
        candidate_1 = os.path.join(input_mask_dir, f"{patient_id}{suffix}")
        
        if os.path.exists(candidate_1):
            input_mask_file = candidate_1
        else:
            print(f"  WARNING: Input mask file not found: {candidate_1}")
            return False
            
    # Check if SUV file exists
    if not os.path.exists(suv_file):
        print(f"  ERROR: SUV file not found: {suv_file}")
        return False
    
    # Check if ground truth exists (optional)
    if not os.path.exists(gt_mask_file):
        print(f"  WARNING: Ground truth file not found: {gt_mask_file}")
        gt_mask_file = None
    
    # Load mapping file to get available mask types
    with open(mapping_file, 'r') as f:
        mapping_data = json.load(f)
    
    # Construct organ mask file paths
    organ_mask_files = {}
    for mask_type in mapping_data.keys():
        mask_file_path = os.path.join(organ_mask_dir, f"{patient_id}_400_0000_{mask_type}.nii")
        if os.path.exists(mask_file_path):
            organ_mask_files[mask_type] = mask_file_path
        else:
            print(f"  WARNING: {mask_type} mask file not found: {mask_file_path}")
    
    if not organ_mask_files:
        print(f"  ERROR: No organ mask files found for {patient_id}")
        return False
    
    print(f"\n{'='*80}")
    print(f"Processing patient: {patient_id}")
    print(f"{'='*80}")
    print(f"  SUV file: {suv_file}")
    print(f"  Organ masks: {len(organ_mask_files)} types found")
    print(f"  Ground truth: {gt_mask_file if gt_mask_file else 'Not available'}")
    print(f"  Output JSON: {output_json}")
    print(f"  Output mask: {output_mask}")
    
    try:
        # Run the combined pipeline
        output_data, filtered_mask = run_combined_pipeline(
            suv_file=suv_file,
            organ_mask_files=organ_mask_files,
            mapping_file=mapping_file,
            output_json=output_json,
            output_mask=output_mask,
            exclude_brain=exclude_brain,
            exclude_kidneys=exclude_kidneys,
            exclude_bladder=exclude_bladder,
            gt_mask_file=gt_mask_file,
            input_mask_file=input_mask_file,
            **kwargs
        )
        
        print(f"\n✓ Successfully processed {patient_id}")
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR processing {patient_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def batch_process(
    suv_dir: str,
    organ_mask_dir: str,
    output_dir: str,
    mapping_file: str,
    pattern: str = "*_0001.nii.gz",
    exclude_brain: bool = True,
    exclude_kidneys: bool = False,
    exclude_bladder: bool = True,
    input_mask_dir: str = None,
    input_mask_suffix: str = None,
    **kwargs
):
    """
    Batch process all patients in a directory.
    
    Args:
        suv_dir: Directory containing SUV files
        organ_mask_dir: Directory containing organ mask files
        output_dir: Directory for output files
        mapping_file: Path to organ mapping JSON file
        pattern: Glob pattern for SUV files (default: *_0001.nii.gz)
        exclude_brain: Whether to exclude brain (default: True)
        exclude_kidneys: Whether to exclude kidneys (default: False)
        exclude_bladder: Whether to exclude bladder (default: True)
        **kwargs: Additional parameters to pass to pipeline
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all SUV files
    suv_pattern = os.path.join(suv_dir, pattern)
    suv_files = glob.glob(suv_pattern)
    suv_files.sort()
    
    if not suv_files:
        print(f"ERROR: No SUV files found matching pattern: {suv_pattern}")
        return
    
    print(f"Found {len(suv_files)} SUV files to process")
    print(f"SUV directory: {suv_dir}")
    print(f"Organ mask directory: {organ_mask_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Exclusions: Brain={exclude_brain}, Kidneys={exclude_kidneys}, Bladder={exclude_bladder}")
    
    # Process each patient
    successful = 0
    failed = 0
    
    for suv_file in suv_files:
        patient_id = extract_patient_id(suv_file)
        
        success = process_single_patient(
            patient_id=patient_id,
            suv_dir=suv_dir,
            organ_mask_dir=organ_mask_dir,
            output_dir=output_dir,
            mapping_file=mapping_file,
            exclude_brain=exclude_brain,
            exclude_kidneys=exclude_kidneys,
            exclude_bladder=exclude_bladder,
            input_mask_dir=input_mask_dir,
            input_mask_suffix=input_mask_suffix,
            **kwargs
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total files: {len(suv_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process PET-CT lesion analysis pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--suv_dir", "-s", required=True,
                       help="Directory containing SUV files (pattern: *_0001.nii.gz)")
    parser.add_argument("--organ_mask_dir", "-o", required=True,
                       help="Directory containing organ mask (totalsegmentator) files")
    parser.add_argument("--output_dir", "-d", required=True,
                       help="Output directory for results")
    parser.add_argument("--mapping_file", "-m",
                       default="data/totalsegmentator_index_mapping.json",
                       help="Path to organ mapping JSON file")
    
    # Analysis Mode (Existing Segmentation)
    parser.add_argument("--input_mask_dir", default=None,
                       help="Directory containing existing lesion masks (enables Analysis Mode)")
    parser.add_argument("--input_mask_suffix", default="_lesion_mask.nii.gz",
                       help="Suffix for existing lesion mask files (default: _lesion_mask.nii.gz)")
    
    # Optional arguments
    parser.add_argument("--pattern", default="*_0001.nii.gz",
                       help="Glob pattern for SUV files")
    
    # Exclusion options
    parser.add_argument("--exclude_brain", type=lambda x: x.lower() in ['true', '1', 'yes'],
                       default=True, help="Exclude brain from lesion detection (default: True)")
    parser.add_argument("--exclude_kidneys", type=lambda x: x.lower() in ['true', '1', 'yes'],
                       default=False, help="Exclude kidneys from lesion detection (default: False)")
    parser.add_argument("--exclude_bladder", type=lambda x: x.lower() in ['true', '1', 'yes'],
                       default=True, help="Exclude bladder from lesion detection (default: True)")
    
    # Pipeline parameters
    parser.add_argument("--suv_threshold", "-t", type=float, default=3.5,
                       help="Minimum SUV value for lesion detection")
    parser.add_argument("--min_voxels", "-p", type=int, default=20,
                       help="Minimum number of voxels for lesion detection")
    parser.add_argument("--connectivity", type=int, default=18, choices=[6, 18, 26],
                       help="Connectivity for lesion segmentation")
    
    # Post-processing parameters
    parser.add_argument("--y_threshold", type=float, default=2.0,
                       help="Y coordinate difference threshold for symmetric detection")
    parser.add_argument("--z_threshold", type=float, default=1.0,
                       help="Z coordinate difference threshold for symmetric detection")
    parser.add_argument("--x_sum_threshold", type=float, default=3.0,
                       help="X coordinate sum threshold for bilateral symmetry")
    parser.add_argument("--overlap_threshold", type=float, default=30.0,
                       help="Organ overlap threshold for Rule 3 exclusion")
    
    # Ground truth validation
    parser.add_argument("--gt_overlap_threshold", type=float, default=0.8,
                       help="Overlap threshold for ground truth matching")
    
    args = parser.parse_args()
    
    # Check if directories exist
    if not os.path.exists(args.suv_dir):
        print(f"ERROR: SUV directory not found: {args.suv_dir}")
        return
    
    if not os.path.exists(args.organ_mask_dir):
        print(f"ERROR: Organ mask directory not found: {args.organ_mask_dir}")
        return
    
    if not os.path.exists(args.mapping_file):
        print(f"ERROR: Mapping file not found: {args.mapping_file}")
        return
    
    # Run batch processing
    batch_process(
        suv_dir=args.suv_dir,
        organ_mask_dir=args.organ_mask_dir,
        output_dir=args.output_dir,
        mapping_file=args.mapping_file,
        pattern=args.pattern,
        exclude_brain=args.exclude_brain,
        exclude_kidneys=args.exclude_kidneys,
        exclude_bladder=args.exclude_bladder,
        suv_threshold=args.suv_threshold,
        min_voxels=args.min_voxels,
        connectivity=args.connectivity,
        y_threshold=args.y_threshold,
        z_threshold=args.z_threshold,
        x_sum_threshold=args.x_sum_threshold,
        overlap_threshold=args.overlap_threshold,
        gt_overlap_threshold=args.gt_overlap_threshold,
        input_mask_dir=args.input_mask_dir,
        input_mask_suffix=args.input_mask_suffix
    )


if __name__ == "__main__":
    main()
