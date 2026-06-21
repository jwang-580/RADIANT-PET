#!/usr/bin/env python3
"""
Add ground truth annotations to lesion description JSON files.

Reads ground truth lesion IDs from CSV file and annotates each lesion
OR uses overlap with ground truth segmentation to annotate each lesion
in the JSON files with an 'is_true_lesion' field.
"""

import os
import sys
import json
import argparse
import pandas as pd
import glob
from pathlib import Path
from typing import Dict, Set, Optional, Tuple, List
import numpy as np
import nibabel as nib

def load_ground_truth_csv(csv_file: str) -> Tuple[Dict[str, Set[int]], Dict[str, Set[int]], Dict[str, str]]:
    """
    Load ground truth CSV and create mapping of study ID to true/false lesion IDs and reports.
    
    Args:
        csv_file: Path to CSV file with columns: study, report, false lesion number, true lesion number
        
    Returns:
        Tuple of:
        - Dictionary mapping study ID to set of explicitly true lesion IDs
        - Dictionary mapping study ID to set of false lesion IDs
        - Dictionary mapping study ID to report text
    """
    print(f"Loading ground truth from: {csv_file}")
    
    df = pd.read_csv(csv_file)
    true_lesions_dict = {}
    false_lesions_dict = {}
    reports = {}
    
    for _, row in df.iterrows():
        study_id = str(row['study']).strip()
        true_lesions_str = str(row['true lesion number']).strip() if 'true lesion number' in row else ''
        false_lesions_str = str(row['false lesion number']).strip() if 'false lesion number' in row else ''
        report_text = str(row['report']).strip() if 'report' in row and not pd.isna(row['report']) else ''
        
        # Store report text
        reports[study_id] = report_text
        
        # Parse true lesion IDs
        if pd.isna(true_lesions_str) or true_lesions_str == '' or true_lesions_str == 'nan':
            true_lesions_dict[study_id] = set()
        else:
            try:
                lesion_ids = [int(x.strip()) for x in true_lesions_str.split(',') if x.strip()]
                true_lesions_dict[study_id] = set(lesion_ids)
            except ValueError as e:
                print(f"Warning: Could not parse true lesion IDs for {study_id}: {true_lesions_str}")
                true_lesions_dict[study_id] = set()
        
        # Parse false lesion IDs
        if pd.isna(false_lesions_str) or false_lesions_str == '' or false_lesions_str == 'nan':
            false_lesions_dict[study_id] = set()
        else:
            try:
                lesion_ids = [int(x.strip()) for x in false_lesions_str.split(',') if x.strip()]
                false_lesions_dict[study_id] = set(lesion_ids)
            except ValueError as e:
                print(f"Warning: Could not parse false lesion IDs for {study_id}: {false_lesions_str}")
                false_lesions_dict[study_id] = set()
    
    print(f"Loaded ground truth for {len(true_lesions_dict)} studies")
    return true_lesions_dict, false_lesions_dict, reports

def extract_study_id(filename: str) -> Optional[str]:
    """
    Extract study ID from filename.
    
    Examples:
        OSU100_acc0_description.json -> OSU100
        OSU22_acc1_description.json -> OSU22
        
    Args:
        filename: JSON filename
        
    Returns:
        Study ID or None if pattern doesn't match
    """
    basename = os.path.basename(filename)
    
    # Remove extension
    if basename.endswith('.json'):
        basename = basename[:-5]
    
    # Remove _description or _results suffix if present
    if basename.endswith('_description'):
        basename = basename[:-12]
    elif basename.endswith('_results'):
        basename = basename[:-8]
    
    # Return the remaining part (OSUxx_accx)
    return basename

def annotate_json_file(json_file: str, true_lesions: Dict[str, Set[int]], 
                       false_lesions: Dict[str, Set[int]],
                       reports: Dict[str, str],
                       output_file: Optional[str] = None) -> bool:
    """
    Annotate a single JSON file with ground truth information.
    
    Logic:
    - If lesion ID is in true_lesions: is_true_lesion = True
    - If lesion ID is in false_lesions: is_true_lesion = False
    - If lesion ID is in neither: is_true_lesion = True (default)
    
    Args:
        json_file: Path to input JSON file
        true_lesions: Dictionary mapping study ID to explicitly true lesion IDs
        false_lesions: Dictionary mapping study ID to false lesion IDs
        reports: Dictionary mapping study ID to report text
        output_file: Optional output path (if None, overwrites input)
        
    Returns:
        True if successful, False otherwise
    """
    # Extract study ID from filename
    study_id = extract_study_id(json_file)
    if study_id is None:
        print(f"Warning: Could not extract study ID from {json_file}")
        return False
    
    # Check if study has ground truth record with actual lesion IDs
    if study_id not in true_lesions and study_id not in false_lesions:
        print(f"Skipping {os.path.basename(json_file)} - Study {study_id} not found in ground truth CSV")
        return False
    
    # Get true and false lesion IDs for this study
    true_lesion_ids = true_lesions.get(study_id, set())
    false_lesion_ids = false_lesions.get(study_id, set())
    
    # Skip if no lesions are defined at all (neither true nor false)
    if not true_lesion_ids and not false_lesion_ids:
        print(f"Skipping {os.path.basename(json_file)} - Study {study_id} has no lesion IDs in CSV")
        return False
    
    print(f"Processing {os.path.basename(json_file)} (Study: {study_id})")

    
    # Load JSON
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return False
    
    print(f"  True lesions: {len(true_lesion_ids)}, False lesions: {len(false_lesion_ids)}")
    
    # Add report to metadata
    report_text = reports.get(study_id, '')
    if 'metadata' not in data:
        data['metadata'] = {}
    data['metadata']['pet_report'] = report_text
    data['metadata']['ground_truth'] = {
        'true_lesion_count': len(true_lesion_ids),
        'true_lesion_ids': sorted(list(true_lesion_ids)),
        'false_lesion_count': len(false_lesion_ids),
        'false_lesion_ids': sorted(list(false_lesion_ids))
    }
    
    # Annotate each lesion
    if 'lesions' not in data:
        print(f"  Warning: No 'lesions' field found in {json_file}")
        return False
    
    true_count = 0
    false_count = 0
    for lesion in data['lesions']:
        lesion_id = lesion.get('id')
        if lesion_id is not None:
            # Logic: explicitly true > explicitly false > default true
            if lesion_id in true_lesion_ids:
                is_true = True
            elif lesion_id in false_lesion_ids:
                is_true = False
            else:
                # Not in either list - default to true
                is_true = True
            
            lesion['is_true_lesion'] = is_true
            if is_true:
                true_count += 1
            else:
                false_count += 1
        else:
            lesion['is_true_lesion'] = True  # Default to true if no ID
    
    print(f"  Marked {true_count} as true, {false_count} as false (out of {len(data['lesions'])} total)")
    
    # Save output
    output_path = output_file if output_file else json_file
    try:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Saved to {output_path}")
        return True
    except Exception as e:
        print(f"  Error saving {output_path}: {e}")
        return False

def process_batch(json_dir: str, true_lesions: Dict[str, Set[int]], 
                  false_lesions: Dict[str, Set[int]],
                  reports: Dict[str, str],
                  output_dir: Optional[str] = None) -> None:
    """
    Process all JSON files in a directory.
    
    Args:
        json_dir: Directory containing JSON files
        true_lesions: Dictionary mapping study ID to explicitly true lesion IDs
        false_lesions: Dictionary mapping study ID to false lesion IDs
        reports: Dictionary mapping study ID to report text
        output_dir: Optional output directory (if None, overwrites in place)
    """
    # Find all description JSON files
    json_files = sorted(glob.glob(os.path.join(json_dir, "*_description.json")))
    
    if not json_files:
        json_files = sorted(glob.glob(os.path.join(json_dir, "*_results.json")))
        
    if not json_files:
        print(f"No *_description.json or *_results.json files found in {json_dir}")
        return
    
    print(f"\nFound {len(json_files)} JSON files to process")
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    for json_file in json_files:
        output_file = None
        if output_dir:
            basename = os.path.basename(json_file)
            output_file = os.path.join(output_dir, basename)
        
        if annotate_json_file(json_file, true_lesions, false_lesions, reports, output_file):
            success_count += 1
        print()
    
    print(f"Successfully processed {success_count}/{len(json_files)} files")

def process_batch_segmentation(json_dir: str, test_seg_dir: str, gt_seg_dir: str, 
                             reports: Dict[str, str] = None, output_dir: Optional[str] = None) -> None:
    """
    Batch process JSON files using segmentation overlap method.
    
    Args:
        json_dir: Directory containing JSON files
        test_seg_dir: Directory containing test segmentation files
        gt_seg_dir: Directory containing ground truth segmentation files
        reports: Optional reports dictionary
        output_dir: Optional output directory
    """
    # Find JSON files
    json_files = sorted(glob.glob(os.path.join(json_dir, "*_description.json")))
    if not json_files:
        json_files = sorted(glob.glob(os.path.join(json_dir, "*_results.json")))
        
    if not json_files:
        print(f"No *_description.json or *_results.json files found in {json_dir}")
        return
        
    print(f"\nFound {len(json_files)} JSON files to process in batch mode")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    success_count = 0
    for json_file in json_files:
        # Derive scan/study ID from JSON filename
        basename = os.path.basename(json_file)
        
        # Strip suffix to get base ID (e.g. OSU100_acc0)
        # Assuming format [ID]_description.json or [ID]_results.json
        if basename.endswith('_description.json'):
            scan_id = basename[:-17]
        elif basename.endswith('_results.json'):
            scan_id = basename[:-13]
        elif basename.endswith('.json'):
            scan_id = basename[:-5]
        else:
            scan_id = basename
            
        # Construct segmentation paths
        # Assumption: NIfTI files are named:
        test_matches = sorted(
            glob.glob(os.path.join(test_seg_dir, f"{scan_id}_*_watershed.nii.gz")) +
            glob.glob(os.path.join(test_seg_dir, f"{scan_id}_mask.nii"))
        )
        gt_matches   = sorted(glob.glob(os.path.join(gt_seg_dir,   f"{scan_id}_gt_dilate.nii.gz")))

        if not test_matches:
            print(f"Skipping {scan_id} - Test seg not found")
            continue

        if not gt_matches:
            print(f"Skipping {scan_id} - GT seg not found")
            continue

        if len(test_matches) > 1:
            print(f"Warning: multiple test segs for {scan_id}: {test_matches}")

        if len(gt_matches) > 1:
            print(f"Warning: multiple GT segs for {scan_id}: {gt_matches}")

        test_seg_path = test_matches[0]
        gt_seg_path   = gt_matches[0]
            
        output_file = None
        if output_dir:
            output_file = os.path.join(output_dir, basename)
            
        if calculate_overlap_ground_truth(json_file, test_seg_path, gt_seg_path, reports, output_file):
            success_count += 1
        print()
            
    print(f"Successfully processed {success_count}/{len(json_files)} files")

def calculate_overlap_ground_truth(json_file: str, test_seg_path: str, gt_seg_path: str, 
                                 reports: Dict[str, str] = None, output_file: Optional[str] = None) -> bool:
    """
    Annotate JSON based on overlap with ground truth segmentation.
    
    Args:
        json_file: Path to test JSON file
        test_seg_path: Path to test segmentation file (.nii.gz)
        gt_seg_path: Path to ground truth segmentation file (.nii.gz)
        reports: Optional dictionary of study ID to report text
        output_file: Optional output path
        
    Returns:
        True if successful
    """
    print(f"Processing {os.path.basename(json_file)} using segmentation overlap")
    print(f"  Test Seg: {test_seg_path}")
    print(f"  GT Seg: {gt_seg_path}")

    try:
        # Load JSON
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        # Load NIfTI files
        test_img = nib.load(test_seg_path)
        gt_img = nib.load(gt_seg_path)
        
        test_data = test_img.get_fdata().astype(int)
        gt_data = gt_img.get_fdata().astype(int)
        
        # Check shapes
        if test_data.shape != gt_data.shape:
            print(f"Error: Shape mismatch - Test {test_data.shape} vs GT {gt_data.shape}")
            return False
            
    except Exception as e:
        print(f"Error loading files: {e}")
        return False

    # Get study ID for report lookup
    study_id = extract_study_id(json_file)
    reports = reports or {}
    report_text = ''
    if study_id:
        report_text = reports.get(study_id, '')
        if study_id in reports:
            print(f"  Added report for {study_id}")
            
    if 'metadata' not in data:
        data['metadata'] = {}
    data['metadata']['pet_report'] = report_text

    # Process lesions
    if 'lesions' not in data:
        print("Error: No 'lesions' key in JSON")
        return False

    true_count = 0
    false_count = 0
    true_lesion_ids = []
    false_lesion_ids = []

    # Get GT binary mask
    gt_mask = gt_data > 0

    for lesion in data['lesions']:
        lesion_id = lesion.get('id')
        if lesion_id is None:
            continue
            
        # Create mask for this lesion
        lesion_mask = (test_data == lesion_id)
        lesion_voxels = np.sum(lesion_mask)
        
        if lesion_voxels == 0:
            print(f"  Warning: Lesion ID {lesion_id} not found in test segmentation")
            is_true = False # Default to false if not found
        else:
            # Calculate overlap
            intersection = np.sum(lesion_mask & gt_mask)
            overlap_ratio = intersection / lesion_voxels
            
            is_true = bool(overlap_ratio >= 0.50)
            
            # Update lesion info
            lesion['is_true_lesion'] = is_true
            lesion['overlap_ratio'] = float(overlap_ratio)
            
        if is_true:
            true_count += 1
            true_lesion_ids.append(lesion_id)
        else:
            false_count += 1
            false_lesion_ids.append(lesion_id)
            
    # Update metadata
    if 'metadata' not in data:
        data['metadata'] = {}
        
    data['metadata']['ground_truth'] = {
        'method': 'segmentation_overlap',
        'overlap_threshold': 0.50,
        'gt_segmentation_file': gt_seg_path,
        'true_lesion_count': true_count,
        'true_lesion_ids': sorted(true_lesion_ids),
        'false_lesion_count': false_count,
        'false_lesion_ids': sorted(false_lesion_ids)
    }

    print(f"  Marked {true_count} true, {false_count} false")

    # Save output
    output_path = output_file if output_file else json_file
    try:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Saved to {output_path}")
        return True
    except Exception as e:
        print(f"  Error saving {output_path}: {e}")
        return False

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Add ground truth annotations to lesion description JSON files"
    )
    
    parser.add_argument("--csv_file", "-c", required=True,
                       help="Path to ground truth CSV file")
    parser.add_argument("--json_file", "-j",
                       help="Path to single JSON file to process")
    parser.add_argument("--json_dir", "-d",
                       default="nnunet/nnUNet_raw/Dataset002_test/results",
                       help="Directory containing JSON files (for batch processing)")
    parser.add_argument("--output_dir", "-o",
                       help="Output directory (if not specified, overwrites in place)")
    parser.add_argument("--batch", action="store_true",
                       help="Enable batch processing mode for csv file input")

    # arguments for segmentation-based GT
    parser.add_argument("--gt_seg", default="nnunet/nnUNet_raw/Dataset001_autopet/gt", help="Path to ground truth segmentation file (.nii.gz) or directory (if batch)")
    parser.add_argument("--test_seg", default="nnunet/nnUNet_raw/Dataset001_autopet/labelstr/nnunet_t8", help="Path to test segmentation file (.nii.gz) or directory (if batch)")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Validate arguments
    if not args.json_file and not args.json_dir:
        print("Error: Must specify either --json_file or --json_dir")
        sys.exit(1)
    
    if args.json_file and args.json_dir:
        print("Error: Cannot specify both --json_file and --json_dir")
        sys.exit(1)
    
    # Determine mode
    if args.gt_seg and args.test_seg:
        # Segmentation-based mode
        
        # Optionally load reports if CSV is provided
        reports = {}
        if args.csv_file and os.path.exists(args.csv_file):
            try:
                # We ignore lesion IDs here
                _, _, reports = load_ground_truth_csv(args.csv_file)
            except Exception as e:
                print(f"Warning: Could not load reports from CSV: {e}")

        if args.json_dir:
            # Batch mode
            # In batch mode, gt_seg and test_seg are treated as directories
            if not os.path.isdir(args.gt_seg) and not os.path.isdir(args.test_seg):
                 # Fallback: if they are files, we can't do batch unless we reuse same file (unlikely)
                 pass
            
            process_batch_segmentation(args.json_dir, args.test_seg, args.gt_seg, reports, args.output_dir)
            sys.exit(0)
            
        elif args.json_file:
            # Single file mode
            output_file = None
            if args.output_dir:
                basename = os.path.basename(args.json_file)
                os.makedirs(args.output_dir, exist_ok=True)
                output_file = os.path.join(args.output_dir, basename)
                
            success = calculate_overlap_ground_truth(args.json_file, args.test_seg, args.gt_seg, reports, output_file)
            sys.exit(0 if success else 1)
        else:
            print("Error: Must specify either --json_file or --json_dir when using segmentation usage")
            sys.exit(1)

    # Standard CSV-based mode
    # Load ground truth
    true_lesions, false_lesions, reports = load_ground_truth_csv(args.csv_file)
    
    # Process files
    if args.json_file:
        # Single file mode
        output_file = None
        if args.output_dir:
            basename = os.path.basename(args.json_file)
            os.makedirs(args.output_dir, exist_ok=True)
            output_file = os.path.join(args.output_dir, basename)
        
        success = annotate_json_file(args.json_file, true_lesions, false_lesions, reports, output_file)
        sys.exit(0 if success else 1)
    
    elif args.json_dir:
        # Batch mode
        process_batch(args.json_dir, true_lesions, false_lesions, reports, args.output_dir)
        sys.exit(0)

if __name__ == "__main__":
    main()
