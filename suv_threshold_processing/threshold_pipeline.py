#!/usr/bin/env python3
"""
Combined PET-CT lesion extraction and rule-based post-processing pipeline.

Supports two operational modes:
1. Extraction Mode (default): Generate lesion segmentation from SUV data, then apply post-processing
2. Analysis Mode: Analyze existing lesion segmentation mask, then apply post-processing

Both modes produce:
1. Filtered JSON file with post-processing rules applied
2. Corresponding NIfTI mask file with only filtered lesions
3. Optional ground truth validation results

Post-processing rules applied in both modes:
- Rule 1: Liver/spleen filtering (>90% overlap with low SUV)
- Rule 2: Symmetric lesion detection
- Rule 3: Organ overlap filtering (bladder, kidney, brain)
"""

import nibabel as nib
import numpy as np
import json
import argparse
import os
from typing import Dict, Any, List, Tuple, Optional
from copy import deepcopy

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import threshold candidate generation.
from suv_threshold_processing.threshold_candidates import (
    PETLesionPipeline, run_suv_extraction
)

# Import rule-based post-processing.
from suv_threshold_processing.rule_based_postprocessing import (
    apply_rule1_liver_spleen_filter,
    apply_rule2_symmetric_detection,
    apply_rule3_organ_overlap_filter,
    PostProcessingStats
)


def filter_mask_by_lesion_ids(original_mask: np.ndarray, lesion_id_mapping: Dict[int, int]) -> np.ndarray:
    """
    Filter the lesion mask to only include lesions that passed post-processing rules.
    
    Args:
        original_mask: Original lesion mask with sequential IDs (1 to N)
        lesion_id_mapping: Mapping from original lesion ID to new sequential ID
                          (only includes lesions that passed filtering)
    
    Returns:
        Filtered mask with only kept lesions, relabeled sequentially
    """
    filtered_mask = np.zeros_like(original_mask, dtype=np.int32)
    
    for original_id, new_id in lesion_id_mapping.items():
        lesion_voxels = (original_mask == original_id)
        filtered_mask[lesion_voxels] = new_id
    
    return filtered_mask


def save_filtered_mask(filtered_mask: np.ndarray, output_path: str, 
                       affine: np.ndarray, header) -> None:
    """
    Save the filtered lesion mask as a NIfTI file.
    
    Args:
        filtered_mask: Filtered lesion mask array
        output_path: Path to save the mask
        affine: Affine transformation matrix
        header: NIfTI header
    """
    mask_img = nib.Nifti1Image(filtered_mask, affine, header)
    nib.save(mask_img, output_path)
    print(f"Saved filtered lesion mask to: {output_path}")


def compare_with_ground_truth(generated_mask: np.ndarray, gt_mask_file: str, 
                              overlap_threshold: float = 0.9) -> Tuple[Dict[int, bool], Dict[str, Any]]:
    """
    Compare generated lesion mask with ground truth mask.
    
    For each generated lesion, check if it contains >=90% of any ground truth lesion.
    If yes, mark as "true", otherwise "false".
    
    Args:
        generated_mask: Generated lesion mask with sequential IDs (1 to N)
        gt_mask_file: Path to ground truth mask file
        overlap_threshold: Minimum overlap ratio to consider a match (default: 0.9 = 90%)
        
    Returns:
        Tuple of:
        - Dictionary mapping generated lesion ID to validation status (True/False)
        - Dictionary with validation statistics
    """
    print(f"\nLoading ground truth mask from: {gt_mask_file}")
    
    # Load ground truth mask
    gt_img = nib.load(gt_mask_file)
    gt_mask = gt_img.get_fdata().astype(np.int32)
    
    # Check if shapes match
    if generated_mask.shape != gt_mask.shape:
        print(f"Warning: Shape mismatch - Generated: {generated_mask.shape}, GT: {gt_mask.shape}")
        print("Attempting to proceed with comparison...")
    
    # Get unique labels from both masks
    generated_labels, gen_counts = np.unique(generated_mask, return_counts=True)
    valid_gen_indices = generated_labels > 0
    generated_labels = generated_labels[valid_gen_indices]
    gen_counts = gen_counts[valid_gen_indices]
    gen_volumes = dict(zip(generated_labels, gen_counts))
    
    gt_labels, gt_counts = np.unique(gt_mask, return_counts=True)
    # Filter out background (0)
    valid_gt_indices = gt_labels > 0
    gt_labels = gt_labels[valid_gt_indices]
    gt_counts = gt_counts[valid_gt_indices]
    gt_volumes = dict(zip(gt_labels, gt_counts))
    
    print(f"Generated mask has {len(generated_labels)} lesions")
    print(f"Ground truth mask has {len(gt_labels)} lesions")
    
    # --- Optimized Overlap Calculation ---
    # Flatten arrays to 1D
    gen_flat = generated_mask.ravel()
    gt_flat = gt_mask.ravel()
    
    # Find voxels where both masks have lesions (intersection candidates)
    # We only care where generated_mask > 0 AND gt_mask > 0
    intersection_mask = (gen_flat > 0) & (gt_flat > 0)
    
    if not np.any(intersection_mask):
        print("No overlap found between generated mask and ground truth.")
        validation_results = {int(uid): {'is_valid': False, 'matched_gt_id': None, 'best_overlap_ratio': 0.0} 
                              for uid in generated_labels}
        matched_gt_lesions = set()
    else:
        gen_intersect = gen_flat[intersection_mask]
        gt_intersect = gt_flat[intersection_mask]
        
        # Use a hash trick to count pairs: gen_id * multiplier + gt_id
        # Multiplier must be > max(gt_labels)
        if len(gt_labels) > 0:
            multiplier = int(np.max(gt_labels)) + 1
        else:
            multiplier = 100000 # Fallback
            
        combined_ids = gen_intersect.astype(np.int64) * multiplier + gt_intersect.astype(np.int64)
        unique_pairs, pair_counts = np.unique(combined_ids, return_counts=True)
        
        # Decode pairs
        pair_gen_ids = unique_pairs // multiplier
        pair_gt_ids = unique_pairs % multiplier
        
        # Store overlaps: gen_id -> {gt_id: count}
        overlaps = {}
        for gen_id, gt_id, count in zip(pair_gen_ids, pair_gt_ids, pair_counts):
            if gen_id not in overlaps:
                overlaps[gen_id] = {}
            overlaps[gen_id][gt_id] = count
            
        # Evaluate matches
        validation_results = {}
        matched_gt_lesions = set()
        
        for gen_id in generated_labels:
            is_valid = False
            matched_gt_id = None
            best_overlap_ratio = 0.0
            
            if gen_id in overlaps:
                # Check all GT lesions this generated lesion touches
                for gt_id, overlap_count in overlaps[gen_id].items():
                    gt_vol = gt_volumes.get(gt_id, 0)
                    gen_vol = gen_volumes.get(gen_id, 0)
                    
                    if gt_vol == 0 or gen_vol == 0: continue
                    
                    # Calculate overlap ratio both ways and take the max
                    ratio_gt = overlap_count / gt_vol
                    ratio_gen = overlap_count / gen_vol
                    ratio = max(ratio_gt, ratio_gen)
                    
                    if ratio > best_overlap_ratio:
                        best_overlap_ratio = ratio
                        
                    if ratio >= overlap_threshold:
                        is_valid = True
                        matched_gt_id = int(gt_id)
                        matched_gt_lesions.add(gt_id)
                        # Keep checking to find best ratio or just break? 
                        # Requirement: "check if it contains >=90% of ANY ground truth lesion"
                        # We can break once valid, but maybe we want the BEST match?
                        # The original code broke on first match. Let's stick to that for speed, 
                        # or find best. Finding best is safer for reporting.
                        # Let's find best ratio.
            
            # If we found multiple valid matches, pick the one with highest overlap?
            # Or just mark as valid.
            # Re-evaluating "break" logic: if we found a valid one, we can stop if we don't care about WHICH one is "best" 
            # beyond just passing the threshold.
            # But let's populate matched_gt_id with the one that has max ratio if multiple pass.
            
            validation_results[int(gen_id)] = {
                'is_valid': is_valid,
                'matched_gt_id': matched_gt_id,
                'best_overlap_ratio': float(best_overlap_ratio)
            }

    # Calculate statistics
    true_positives = sum(1 for v in validation_results.values() if v['is_valid'])
    false_positives = len(generated_labels) - true_positives
    false_negatives = len(gt_labels) - len(matched_gt_lesions)
    
    validation_stats = {
        'total_generated_lesions': len(generated_labels),
        'total_gt_lesions': len(gt_labels),
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': true_positives / len(generated_labels) if len(generated_labels) > 0 else 0.0,
        'recall': true_positives / len(gt_labels) if len(gt_labels) > 0 else 0.0,
        'overlap_threshold': overlap_threshold
    }
    
    # Calculate F1 score
    precision = validation_stats['precision']
    recall = validation_stats['recall']
    validation_stats['f1_score'] = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    print(f"\nGround Truth Validation Results:")
    print(f"  True Positives: {true_positives}")
    print(f"  False Positives: {false_positives}")
    print(f"  False Negatives: {false_negatives}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {validation_stats['f1_score']:.3f}")
    
    return validation_results, validation_stats


def run_combined_pipeline(
    suv_file: str,
    organ_mask_files: dict,
    mapping_file: str,
    output_json: str,
    output_mask: str,
    # Input mode
    input_mask_file: Optional[str] = None,
    # SUV extraction parameters
    suv_threshold: float = 2.5,
    min_voxels: int = 20,
    connectivity: int = 18,
    pre_min_voxels: int = 20,
    post_min_voxels: int = None,
    exclude_brain: bool = False,
    exclude_kidneys: bool = False,
    exclude_bladder: bool = False,
    z_boundary_voxels: int = 2,
    # Rule-based post-processing parameters
    y_threshold: float = 2.0,
    z_threshold: float = 1.0,
    x_sum_threshold: float = 3.0,
    suv_pp_threshold: float = 1.0,
    shape_threshold: float = 0.3,
    overlap_threshold: float = 30.0,
    summary_file: str = None,
    # Ground truth comparison
    gt_mask_file: Optional[str] = None,
    gt_overlap_threshold: float = 0.9
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Run the combined pipeline: Lesion analysis + rule-based post-processing + optional ground truth validation.
    
    Supports two modes:
    - Extraction Mode (input_mask_file=None): Generate new segmentation from SUV data
    - Analysis Mode (input_mask_file provided): Analyze existing segmentation mask
    
    Args:
        suv_file: Path to SUV NIfTI file
        organ_mask_files: Dictionary of mask type to file path (e.g., {'total': 'path.nii', 'head_glands': 'path2.nii'})
        mapping_file: Path to organ mapping JSON file
        output_json: Path to save filtered lesion results JSON
        output_mask: Path to save filtered lesion mask NIfTI
        input_mask_file: Optional path to existing lesion mask (enables analysis mode)
        suv_threshold: Minimum SUV value for lesion detection (extraction mode only)
        min_voxels: Minimum number of voxels for lesion detection
        connectivity: Connectivity for lesion segmentation (6, 18, or 26)
        pre_min_voxels: Pre-filter minimum voxels for initial blob cleanup
        post_min_voxels: Post-filter minimum voxels for watershed pieces
        exclude_brain: Whether to exclude brain from lesion detection (extraction mode)
        exclude_kidneys: Whether to exclude kidneys from lesion detection (extraction mode)
        exclude_bladder: Whether to exclude bladder from lesion detection (extraction mode)
        z_boundary_voxels: Number of voxels to exclude from Z-axis boundaries
        y_threshold: Y coordinate difference threshold for symmetric detection
        z_threshold: Z coordinate difference threshold for symmetric detection
        x_sum_threshold: X coordinate sum threshold for bilateral symmetry
        suv_pp_threshold: SUV difference threshold for symmetric detection
        shape_threshold: Shape similarity threshold for symmetric detection
        overlap_threshold: Percentage threshold for organ overlap exclusion (Rule 3)
        summary_file: Optional path to save summary report
        gt_mask_file: Optional path to ground truth mask for validation
        gt_overlap_threshold: Overlap threshold for ground truth matching (default: 0.9)
        
    Returns:
        Tuple of (filtered_output_data, filtered_mask)
        - filtered_output_data: Dictionary with metadata and filtered lesion list
        - filtered_mask: NumPy array with filtered lesion mask
    """
    print("=" * 80)
    print("COMBINED PET-CT LESION EXTRACTION AND POST-PROCESSING PIPELINE")
    print("=" * 80)
    
    # ========== STEP 1: Lesion Data Preparation ==========
    print("\n" + "=" * 80)
    
    if input_mask_file:
        # Analysis mode: Use existing segmentation mask
        print("STEP 1: ANALYZING EXISTING SEGMENTATION MASK")
        print("=" * 80)
        
        print(f"Input files:")
        print(f"  Lesion mask: {input_mask_file}")
        print(f"  SUV file: {suv_file}")
        print(f"  Organ mask files:")
        for mask_type, mask_file in organ_mask_files.items():
            print(f"    {mask_type}: {mask_file}")
        print(f"  Mapping file: {mapping_file}")
        print()
        
        # Import analyze_existing_segmentation function
        from suv_threshold_processing.threshold_candidates import analyze_existing_segmentation
        
        # Load the existing lesion mask
        print("Loading existing lesion segmentation mask...")
        lesion_mask_img = nib.load(input_mask_file)
        original_mask = lesion_mask_img.get_fdata().astype(np.int32)
        
        # Initialize pipeline to get organ data and SUV data
        print("Initializing pipeline for organ and SUV analysis...")
        pipeline = PETLesionPipeline(suv_file, organ_mask_files, mapping_file)
        
        # Import necessary utilities
        from utils.lesion_features import (
            convert_numpy_types, analyze_organ_overlaps, find_closest_organs,
            combine_organ_results, determine_vertebrae_level, compute_shape_metrics
        )
        from suv_threshold_processing.threshold_candidates import Lesion
        
        # Get unique lesion labels
        unique_labels = np.unique(original_mask)
        unique_labels = unique_labels[unique_labels > 0]
        print(f"Found {len(unique_labels)} lesions in the mask")
        
        # Analyze each lesion
        lesions = []
        for lesion_id in unique_labels:
            lesion_voxels = (original_mask == lesion_id)
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
        
        # Build metadata
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
        
        # Store affine and header for later use
        mask_affine = pipeline.suv_img.affine
        mask_header = pipeline.suv_img.header
        
    else:
        # Extraction mode: Generate new segmentation
        print("STEP 1: SUV EXTRACTION")
        print("=" * 80)
        
        # Initialize and run pipeline without saving intermediate files
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
        
        # Handle exclusion settings
        excluded_organs = []
        if exclude_brain: excluded_organs.append("brain")
        if exclude_kidneys: excluded_organs.append("kidneys")
        if exclude_bladder: excluded_organs.append("bladder")
        print(f"  Excluding organs: {', '.join(excluded_organs) if excluded_organs else 'none'}")
        print()
        
        # Initialize pipeline
        pipeline = PETLesionPipeline(suv_file, organ_mask_files, mapping_file)
        
        # Run extraction (without saving mask)
        lesions = pipeline.run_pipeline(
            suv_threshold=suv_threshold,
            min_voxels=min_voxels,
            save_mask_path=None,  # Don't save intermediate mask
            exclude_brain=exclude_brain,
            exclude_kidneys=exclude_kidneys,
            exclude_bladder=exclude_bladder,
            z_boundary_voxels=z_boundary_voxels
        )
        
        # Get the mask directly from the pipeline object
        original_mask = pipeline.final_lesion_mask.copy()
        
        # Build metadata
        pipeline_metadata = {
            'pipeline_version': '1.0',
            'processing_parameters': {
                'suv_threshold': suv_threshold,
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
        
        # Store affine and header for later use
        mask_affine = pipeline.suv_img.affine
        mask_header = pipeline.suv_img.header
    
    # Common processing for both modes
    from utils.lesion_features import convert_numpy_types
    
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
    
    # Create initial data structure
    initial_data = {
        'metadata': pipeline_metadata,
        'lesions': lesion_data
    }
    
    # Convert NumPy types to Python native types
    initial_data = convert_numpy_types(initial_data)
    
    original_lesion_count = len(initial_data['lesions'])
    print(f"\nProcessed {original_lesion_count} initial lesions")
    
    # ========== STEP 2: Rule-Based Post-Processing ==========
    print("\n" + "=" * 80)
    print("STEP 2: RULE-BASED POST-PROCESSING")
    print("=" * 80)
    
    original_lesions = initial_data['lesions']
    
    # Get liver SUV upper bound from metadata
    liver_analysis = initial_data['metadata']['liver_suv_analysis']
    liver_upper_bound = liver_analysis.get('upper_bound', float('inf'))
    print(f"Liver SUV upper bound: {liver_upper_bound}")
    
    # Apply Rule 1: Liver/Spleen filtering
    print("\nApplying Rule 1: Liver/Spleen filtering...")
    filtered_lesions_r1, excluded_count_r1 = apply_rule1_liver_spleen_filter(
        original_lesions, liver_upper_bound
    )
    print(f"Excluded {excluded_count_r1} lesions with Rule 1")
    
    # Apply Rule 2: Symmetric lesion detection
    print("\nApplying Rule 2: Symmetric lesion detection...")
    processed_lesions, symmetric_pairs = apply_rule2_symmetric_detection(
        filtered_lesions_r1, y_threshold, z_threshold, x_sum_threshold, 
        suv_pp_threshold, shape_threshold
    )
    print(f"Found {symmetric_pairs} symmetric lesion pairs")
    
    # Apply Rule 3: Organ overlap filtering
    print("\nApplying Rule 3: Organ overlap filtering...")
    filtered_lesions_r3, excluded_count_r3 = apply_rule3_organ_overlap_filter(
        processed_lesions, overlap_threshold
    )
    print(f"Excluded {excluded_count_r3} lesions with Rule 3")
    
    # ========== STEP 3: Create Filtered Outputs ==========
    print("\n" + "=" * 80)
    print("STEP 3: CREATING FILTERED OUTPUTS")
    print("=" * 80)
    
    # Create lesion ID mapping (original ID -> new sequential ID)
    lesion_id_mapping = {}
    for new_id, lesion in enumerate(filtered_lesions_r3, 1):
        original_id = lesion['id']
        lesion_id_mapping[original_id] = new_id
        # Update lesion ID to new sequential ID
        lesion['id'] = new_id
    
    # Filter the mask to only include kept lesions
    filtered_mask = filter_mask_by_lesion_ids(original_mask, lesion_id_mapping)
    
    # Verify filtered mask
    unique_labels = np.unique(filtered_mask)
    unique_labels = unique_labels[unique_labels > 0]
    print(f"Filtered mask contains {len(unique_labels)} lesion regions")
    
    # Create output data structure
    output_data = deepcopy(initial_data)
    output_data['lesions'] = filtered_lesions_r3
    
    # Add post-processing metadata
    output_data['metadata']['post_processing'] = {
        'rules_applied': ['liver_spleen_filter', 'symmetric_detection', 'organ_overlap_filter'],
        'original_lesion_count': original_lesion_count,
        'rule1_excluded_count': excluded_count_r1,
        'rule2_symmetric_pairs': symmetric_pairs,
        'rule3_excluded_count': excluded_count_r3,
        'final_lesion_count': len(filtered_lesions_r3),
        'parameters': {
            'y_threshold': y_threshold,
            'z_threshold': z_threshold,
            'x_sum_threshold': x_sum_threshold,
            'suv_threshold': suv_pp_threshold,
            'shape_threshold': shape_threshold,
            'overlap_threshold': overlap_threshold
        }
    }
    
    # Update lesion statistics
    output_data['metadata']['lesion_statistics']['total_lesions_found'] = len(filtered_lesions_r3)
    
    # ========== STEP 4: Ground Truth Comparison (Optional) ==========
    if gt_mask_file and os.path.exists(gt_mask_file):
        print("\n" + "=" * 80)
        print("STEP 4: GROUND TRUTH COMPARISON")
        print("=" * 80)
        
        validation_results, validation_stats = compare_with_ground_truth(
            filtered_mask, gt_mask_file, gt_overlap_threshold
        )
        
        # Add validation information to each lesion
        for lesion in output_data['lesions']:
            lesion_id = lesion['id']
            if lesion_id in validation_results:
                val_info = validation_results[lesion_id]
                lesion['ground_truth_validation'] = {
                    'is_true_positive': val_info['is_valid'],
                    'matched_gt_lesion_id': val_info['matched_gt_id'],
                    'best_overlap_ratio': val_info['best_overlap_ratio']
                }
            else:
                lesion['ground_truth_validation'] = {
                    'is_true_positive': False,
                    'matched_gt_lesion_id': None,
                    'best_overlap_ratio': 0.0
                }
        
        # Add validation statistics to metadata
        output_data['metadata']['ground_truth_validation'] = validation_stats
        output_data['metadata']['ground_truth_validation']['gt_mask_file'] = gt_mask_file
    elif gt_mask_file:
        print(f"\nWarning: Ground truth mask file not found: {gt_mask_file}")
        print("Skipping ground truth comparison.")
    else:
        print("\nNo ground truth mask provided. Skipping validation.")
    
    # Save filtered JSON
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved filtered lesion data to: {output_json}")
    
    # Save filtered mask
    save_filtered_mask(filtered_mask, output_mask, mask_affine, mask_header)
    
    # Generate summary report if requested
    if summary_file:
        stats = PostProcessingStats(
            total_lesions=original_lesion_count,
            rule1_excluded=excluded_count_r1,
            rule2_symmetric_pairs=symmetric_pairs,
            rule3_excluded=excluded_count_r3,
            final_lesions=len(filtered_lesions_r3)
        )
        
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("COMBINED PIPELINE SUMMARY")
        summary_lines.append("=" * 80)
        summary_lines.append(f"\nOriginal lesions extracted: {stats.total_lesions}")
        summary_lines.append(f"Rule 1 excluded (liver/spleen): {stats.rule1_excluded}")
        summary_lines.append(f"Rule 3 excluded (organ overlap): {stats.rule3_excluded}")
        summary_lines.append(f"Symmetric pairs found: {stats.rule2_symmetric_pairs}")
        summary_lines.append(f"Final lesions: {stats.final_lesions}")
        summary_lines.append(f"\nLiver SUV upper bound: {liver_upper_bound}")
        summary_lines.append(f"\nOutput files:")
        summary_lines.append(f"  JSON: {output_json}")
        summary_lines.append(f"  Mask: {output_mask}")
        
        summary_text = "\n".join(summary_lines)
        with open(summary_file, 'w') as f:
            f.write(summary_text)
        print(f"Summary report saved to: {summary_file}")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Original lesions: {original_lesion_count}")
    print(f"Excluded by Rule 1: {excluded_count_r1}")
    print(f"Excluded by Rule 3: {excluded_count_r3}")
    print(f"Symmetric pairs found: {symmetric_pairs}")
    print(f"Final lesions: {len(filtered_lesions_r3)}")
    print(f"\nOutput files:")
    print(f"  Filtered JSON: {output_json}")
    print(f"  Filtered Mask: {output_mask}")
    
    return output_data, filtered_mask


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Combined PET Lesion Extraction and Rule-Based Post-Processing Pipeline",
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
                       help="Prefix for organ mask files")
    parser.add_argument("--mapping_file", "-m",
                       default="data/totalsegmentator_index_mapping.json",
                       help="Path to organ mapping JSON file")
    
    # Output files
    parser.add_argument("--output_json", "-j", default="lesion_results_filtered.json",
                       help="Output JSON file for filtered lesion results")
    parser.add_argument("--output_mask", "-k", default="lesion_mask_filtered.nii",
                       help="Output NIfTI file for filtered lesion mask")
    parser.add_argument("--summary", default=None,
                       help="Optional summary report file")
    
    # SUV extraction parameters
    parser.add_argument("--suv_threshold", "-t", type=float, default=3.5,
                       help="Minimum SUV value for lesion detection")
    parser.add_argument("--min_voxels", "-p", type=int, default=20,
                       help="Minimum number of voxels for lesion detection")
    parser.add_argument("--connectivity", type=int, default=18, choices=[6, 18, 26],
                       help="Connectivity for lesion segmentation")
    parser.add_argument("--pre_min_voxels", type=int, default=20,
                       help="Pre-filter minimum voxels")
    parser.add_argument("--post_min_voxels", type=int, default=None,
                       help="Post-filter minimum voxels")
    
    # Exclusion options
    parser.add_argument("--exclude_brain", type=lambda x: x.lower() in ['true', '1', 'yes'], 
                       default=False, help="Exclude brain from lesion detection")
    parser.add_argument("--exclude_kidneys", type=lambda x: x.lower() in ['true', '1', 'yes'], 
                       default=False, help="Exclude kidneys from lesion detection")
    parser.add_argument("--exclude_bladder", type=lambda x: x.lower() in ['true', '1', 'yes'], 
                       default=False, help="Exclude bladder from lesion detection")
    parser.add_argument("--z_boundary_voxels", type=int, default=2,
                       help="Number of voxels to exclude from Z-axis boundaries")
    
    # Rule-based post-processing parameters
    parser.add_argument("--y_threshold", type=float, default=2.0,
                       help="Y coordinate difference threshold for symmetric detection")
    parser.add_argument("--z_threshold", type=float, default=1.0,
                       help="Z coordinate difference threshold for symmetric detection")
    parser.add_argument("--x_sum_threshold", type=float, default=3.0,
                       help="X coordinate sum threshold for bilateral symmetry")
    parser.add_argument("--suv_pp_threshold", type=float, default=1.0,
                       help="SUV difference threshold for symmetric detection")
    parser.add_argument("--shape_threshold", type=float, default=0.3,
                       help="Shape similarity threshold for symmetric detection")
    parser.add_argument("--overlap_threshold", type=float, default=30.0,
                       help="Organ overlap threshold for Rule 3 exclusion")
    
    # Ground truth comparison
    parser.add_argument("--gt_mask_file", default=None,
                       help="Optional ground truth mask file for validation")
    parser.add_argument("--gt_overlap_threshold", type=float, default=0.8,
                       help="Overlap threshold for ground truth matching (default: 0.8 = 80%)")
    
    return parser.parse_args()


def main():
    """Main function to run the combined pipeline"""
    args = parse_arguments()
    
    # Check if mapping file exists
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
    
    # Run combined pipeline
    output_data, filtered_mask = run_combined_pipeline(
        suv_file=args.suv_file,
        organ_mask_files=organ_mask_files,
        mapping_file=args.mapping_file,
        output_json=args.output_json,
        output_mask=args.output_mask,
        input_mask_file=args.input_mask_file,
        suv_threshold=args.suv_threshold,
        min_voxels=args.min_voxels,
        connectivity=args.connectivity,
        pre_min_voxels=args.pre_min_voxels,
        post_min_voxels=args.post_min_voxels,
        exclude_brain=args.exclude_brain,
        exclude_kidneys=args.exclude_kidneys,
        exclude_bladder=args.exclude_bladder,
        z_boundary_voxels=args.z_boundary_voxels,
        y_threshold=args.y_threshold,
        z_threshold=args.z_threshold,
        x_sum_threshold=args.x_sum_threshold,
        suv_pp_threshold=args.suv_pp_threshold,
        shape_threshold=args.shape_threshold,
        overlap_threshold=args.overlap_threshold,
        summary_file=args.summary,
        gt_mask_file=args.gt_mask_file,
        gt_overlap_threshold=args.gt_overlap_threshold
    )


if __name__ == "__main__":
    main()
