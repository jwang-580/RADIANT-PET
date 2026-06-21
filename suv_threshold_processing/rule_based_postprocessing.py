#!/usr/bin/env python3
"""
Rule-based post-processing for PET-CT lesion analysis.
Works purely based on lesion JSON file from pipeline.py.

Rules implemented:
1. Exclude lesions in liver and spleen with >90% overlap and mean SUV < liver upper bound
2. Find symmetric lesions and add symmetry tags based on location, SUV, and shape
3. Exclude lesions with >30% overlap with bladder, kidney, or brain
"""

import json
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple
import math
from dataclasses import dataclass
from copy import deepcopy


@dataclass
class PostProcessingStats:
    """Statistics from post-processing operations"""
    total_lesions: int
    rule1_excluded: int
    rule2_symmetric_pairs: int
    rule3_excluded: int
    final_lesions: int


def load_lesion_data(json_file: str) -> Dict[str, Any]:
    """Load lesion data from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def save_processed_data(data: Dict[str, Any], output_file: str) -> None:
    """Save processed lesion data to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)


def apply_rule1_liver_spleen_filter(lesions: List[Dict], liver_upper_bound: float) -> Tuple[List[Dict], int]:
    """
    Rule 1: Exclude lesions in liver and spleen with >90% overlap and mean SUV < liver upper bound
    
    Args:
        lesions: List of lesion dictionaries
        liver_upper_bound: Upper bound SUV threshold from liver analysis
        
    Returns:
        Tuple of (filtered_lesions, excluded_count)
    """
    filtered_lesions = []
    excluded_count = 0
    
    for lesion in lesions:
        should_exclude = False
        organ_overlaps = lesion.get('organ_overlaps', {})
        mean_suv = lesion.get('mean_suv', 0)
        
        # Check liver overlap (parse percentage string)
        liver_overlap_str = organ_overlaps.get('liver', '0%')
        liver_overlap = float(liver_overlap_str.rstrip('%')) if liver_overlap_str != '0%' else 0.0
        if liver_overlap > 90 and mean_suv < liver_upper_bound:
            should_exclude = True
            lesion['exclusion_reason'] = f"Liver overlap {liver_overlap:.1f}% with SUV {mean_suv:.2f} < {liver_upper_bound:.2f}"
        
        # Check spleen overlap (parse percentage string)
        spleen_overlap_str = organ_overlaps.get('spleen', '0%')
        spleen_overlap = float(spleen_overlap_str.rstrip('%')) if spleen_overlap_str != '0%' else 0.0
        if spleen_overlap > 90 and mean_suv < liver_upper_bound:
            should_exclude = True
            lesion['exclusion_reason'] = f"Spleen overlap {spleen_overlap:.1f}% with SUV {mean_suv:.2f} < {liver_upper_bound:.2f}"
        
        if should_exclude:
            excluded_count += 1
        else:
            filtered_lesions.append(lesion)
    
    return filtered_lesions, excluded_count


def apply_rule3_organ_overlap_filter(lesions: List[Dict], overlap_threshold: float = 30.0) -> Tuple[List[Dict], int]:
    """
    Rule 3: Exclude lesions with >30% overlap with bladder, kidney, or brain
    
    Args:
        lesions: List of lesion dictionaries
        overlap_threshold: Percentage threshold for organ overlap exclusion (default: 30.0)
        
    Returns:
        Tuple of (filtered_lesions, excluded_count)
    """
    filtered_lesions = []
    excluded_count = 0
    
    for lesion in lesions:
        should_exclude = False
        organ_overlaps = lesion.get('organ_overlaps', {})
        exclusion_reasons = []
        
        # Check overlap with exclusion organs: bladder, kidney, brain
        exclusion_organs = ['urinary_bladder', 'kidney_left', 'kidney_right', 'brain']
        
        for organ in exclusion_organs:
            overlap_str = organ_overlaps.get(organ, '0%')
            overlap = float(overlap_str.rstrip('%')) if overlap_str != '0%' else 0.0
            
            if overlap > overlap_threshold:
                should_exclude = True
                exclusion_reasons.append(f"{organ} overlap {overlap:.1f}%")
        
        if should_exclude:
            excluded_count += 1
        else:
            filtered_lesions.append(lesion)
    
    return filtered_lesions, excluded_count


def are_lesions_symmetric(lesion1: Dict, lesion2: Dict, 
                         y_threshold: float = 2.0,
                         z_threshold: float = 1.0,
                         x_sum_threshold: float = 3.0,
                         suv_difference_threshold: float = 1.0,
                         shape_similarity_threshold: float = 0.3) -> bool:
    """
    Determine if two lesions are symmetric based on bilateral anatomy criteria
    
    For bilateral symmetry:
    - Close z coordinates (similar axial level)
    - Close y coordinates (similar anterior-posterior position)  
    - Opposite x coordinates where x1 + x2 ≈ 0 (bilateral across midline)
    
    Args:
        lesion1, lesion2: Lesion dictionaries
        y_threshold: Maximum difference in y coordinates (anterior-posterior)
        z_threshold: Maximum difference in z coordinates (superior-inferior)
        x_sum_threshold: Maximum absolute value of x1 + x2 (bilateral symmetry)
        suv_difference_threshold: Maximum SUV difference for symmetry
        shape_similarity_threshold: Maximum shape metric difference for symmetry
        
    Returns:
        True if lesions are considered symmetric
    """
    # Get coordinates [x, y, z]
    coords1 = lesion1.get('center_coords', [0, 0, 0])
    coords2 = lesion2.get('center_coords', [0, 0, 0])
    
    x1, y1, z1 = coords1
    x2, y2, z2 = coords2
    
    # Check bilateral symmetry: x1 + x2 should be close to 0
    x_sum = x1 + x2
    if abs(x_sum) > x_sum_threshold:
        return False
    
    # Check similar y coordinates (anterior-posterior position)
    y_diff = abs(y1 - y2)
    if y_diff > y_threshold:
        return False
    
    # Check similar z coordinates (axial level)
    z_diff = abs(z1 - z2)
    if z_diff > z_threshold:
        return False
    
    # Check SUV similarity
    suv1 = lesion1.get('mean_suv', 0)
    suv2 = lesion2.get('mean_suv', 0)
    suv_diff = abs(suv1 - suv2)
    if suv_diff > suv_difference_threshold:
        return False
    
    # Check shape similarity if shape data is available
    shape1 = lesion1.get('shape')
    shape2 = lesion2.get('shape')
    
    if shape1 and shape2:
        # Compare key shape metrics
        shape_metrics = ['elongation', 'flatness', 'solidity']
        for metric in shape_metrics:
            val1 = shape1.get(metric, 0)
            val2 = shape2.get(metric, 0)
            if abs(val1 - val2) > shape_similarity_threshold:
                return False
    
    return True


def apply_rule2_symmetric_detection(lesions: List[Dict], y_threshold: float = 15.0, z_threshold: float = 15.0, x_sum_threshold: float = 10.0, suv_threshold: float = 1.0, shape_threshold: float = 0.3) -> Tuple[List[Dict], int]:
    """
    Rule 2: Find symmetric lesions and add symmetry tags
    
    Args:
        lesions: List of lesion dictionaries
        
    Returns:
        Tuple of (lesions_with_symmetry_tags, symmetric_pairs_count)
    """
    lesions_with_tags = deepcopy(lesions)
    symmetric_pairs = 0
    processed_ids = set()
    
    # Initialize symmetry fields
    for lesion in lesions_with_tags:
        lesion['is_symmetric'] = False
        lesion['symmetric_partner_id'] = None
        lesion['symmetry_score'] = 0.0
    
    # Compare each lesion with every other lesion
    for i, lesion1 in enumerate(lesions_with_tags):
        if lesion1['id'] in processed_ids:
            continue
            
        for j, lesion2 in enumerate(lesions_with_tags[i+1:], i+1):
            if lesion2['id'] in processed_ids:
                continue
                
            if are_lesions_symmetric(lesion1, lesion2, y_threshold, z_threshold, x_sum_threshold, suv_threshold, shape_threshold):
                # Mark both lesions as symmetric
                lesion1['is_symmetric'] = True
                lesion2['is_symmetric'] = True
                lesion1['symmetric_partner_id'] = lesion2['id']
                lesion2['symmetric_partner_id'] = lesion1['id']
                
                # Calculate symmetry score based on coordinate alignment and SUV similarity
                coords1 = lesion1['center_coords']
                coords2 = lesion2['center_coords']
                x1, y1, z1 = coords1
                x2, y2, z2 = coords2
                
                # Score components: better symmetry = lower values
                x_sum_component = abs(x1 + x2) / x_sum_threshold  # How close to bilateral symmetry
                y_diff_component = abs(y1 - y2) / y_threshold     # How similar y positions
                z_diff_component = abs(z1 - z2) / z_threshold     # How similar z positions 
                suv_diff_component = abs(lesion1['mean_suv'] - lesion2['mean_suv']) / suv_threshold
                
                # Higher score = better symmetry (lower normalized differences)
                total_component = x_sum_component + y_diff_component + z_diff_component + suv_diff_component
                symmetry_score = 1.0 / (1.0 + total_component)
                lesion1['symmetry_score'] = symmetry_score
                lesion2['symmetry_score'] = symmetry_score
                
                processed_ids.add(lesion1['id'])
                processed_ids.add(lesion2['id'])
                symmetric_pairs += 1
                break  # Each lesion can only have one symmetric partner
    
    return lesions_with_tags, symmetric_pairs


def generate_summary_report(original_data: Dict, processed_data: Dict, stats: PostProcessingStats) -> str:
    """Generate a summary report of post-processing results"""
    report = []
    report.append("=== LESION POST-PROCESSING SUMMARY ===")
    report.append(f"Original lesions: {stats.total_lesions}")
    report.append(f"Rule 1 excluded: {stats.rule1_excluded}")
    report.append(f"Rule 3 excluded: {stats.rule3_excluded}")
    report.append(f"Symmetric pairs found: {stats.rule2_symmetric_pairs}")
    report.append(f"Final lesions: {stats.final_lesions}")
    report.append("")
    
    # Liver analysis info
    liver_analysis = original_data['metadata']['liver_suv_analysis']
    report.append(f"Liver SUV upper bound: {liver_analysis.get('upper_bound', 'N/A')}")
    report.append("")
    
    # Excluded lesions details
    if stats.rule1_excluded > 0:
        report.append("=== EXCLUDED LESIONS (Rule 1) ===")
        for lesion in original_data['lesions']:
            if lesion.get('excluded_by_rule1', False):
                report.append(f"Lesion {lesion['id']}: {lesion.get('exclusion_reason', 'Unknown reason')}")
        report.append("")
    
    if stats.rule3_excluded > 0:
        report.append("=== EXCLUDED LESIONS (Rule 3) ===")
        for lesion in original_data['lesions']:
            if lesion.get('excluded_by_rule3', False):
                report.append(f"Lesion {lesion['id']}: {lesion.get('exclusion_reason', 'Unknown reason')}")
        report.append("")
    
    # Symmetric pairs details
    if stats.rule2_symmetric_pairs > 0:
        report.append("=== SYMMETRIC LESION PAIRS (Rule 2) ===")
        processed_ids = set()
        for lesion in processed_data['lesions']:
            if lesion.get('is_symmetric', False) and lesion['id'] not in processed_ids:
                partner_id = lesion.get('symmetric_partner_id')
                score = lesion.get('symmetry_score', 0)
                report.append(f"Symmetric pair: Lesion {lesion['id']} ↔ Lesion {partner_id} (score: {score:.3f})")
                processed_ids.add(lesion['id'])
                processed_ids.add(partner_id)
        report.append("")
    
    return "\n".join(report)


def run_rule_based_processing(input_json: str, output_json: str, 
                             y_threshold: float = 2.0, z_threshold: float = 1.0,
                             x_sum_threshold: float = 3.0, suv_threshold: float = 1.0,
                             shape_threshold: float = 0.3, overlap_threshold: float = 30.0,
                             summary_file: str = None) -> dict:
    """
    Core rule-based post-processing function without CLI parsing
    
    Args:
        input_json: Input JSON file path
        output_json: Output JSON file path  
        y_threshold: Y coordinate difference threshold for symmetric lesion detection
        z_threshold: Z coordinate difference threshold for symmetric lesion detection
        x_sum_threshold: X coordinate sum threshold for bilateral symmetry
        suv_threshold: SUV difference threshold for symmetric lesion detection
        shape_threshold: Shape similarity threshold for symmetric lesion detection
        overlap_threshold: Percentage threshold for organ overlap exclusion (Rule 3)
        summary_file: Optional summary report file path
        
    Returns:
        Dictionary containing the processed output data
    """
    # Load input data
    print(f"Loading lesion data from: {input_json}")
    data = load_lesion_data(input_json)
    
    original_lesions = data['lesions']
    original_count = len(original_lesions)
    
    # Get liver SUV upper bound from metadata
    liver_analysis = data['metadata']['liver_suv_analysis']
    liver_upper_bound = liver_analysis.get('upper_bound', float('inf'))
    
    print(f"Original lesions: {original_count}")
    print(f"Liver SUV upper bound: {liver_upper_bound}")
    
    # Apply Rule 1: Liver/Spleen filtering
    print("\nApplying Rule 1: Liver/Spleen filtering...")
    filtered_lesions_r1, excluded_count_r1 = apply_rule1_liver_spleen_filter(original_lesions, liver_upper_bound)
    print(f"Excluded {excluded_count_r1} lesions with Rule 1")

    # Apply Rule 2: Symmetric lesion detection
    print("\nApplying Rule 2: Symmetric lesion detection...")
    processed_lesions, symmetric_pairs = apply_rule2_symmetric_detection(
        filtered_lesions_r1, y_threshold, z_threshold, x_sum_threshold, suv_threshold, shape_threshold)
    print(f"Found {symmetric_pairs} symmetric lesion pairs")
    
    # Apply Rule 3: Organ overlap filtering (after symmetry detection)
    print("\nApplying Rule 3: Organ overlap filtering...")
    filtered_lesions_r3, excluded_count_r3 = apply_rule3_organ_overlap_filter(processed_lesions, overlap_threshold)
    print(f"Excluded {excluded_count_r3} lesions with Rule 3")
    
    # Create output data structure
    output_data = deepcopy(data)
    output_data['lesions'] = filtered_lesions_r3
    
    # Add post-processing metadata
    output_data['metadata']['post_processing'] = {
        'rules_applied': ['liver_spleen_filter', 'symmetric_detection', 'organ_overlap_filter'],
        'original_lesion_count': original_count,
        'rule1_excluded_count': excluded_count_r1,
        'rule2_symmetric_pairs': symmetric_pairs,
        'rule3_excluded_count': excluded_count_r3,
        'final_lesion_count': len(filtered_lesions_r3),
        'parameters': {
            'y_threshold': y_threshold,
            'z_threshold': z_threshold, 
            'x_sum_threshold': x_sum_threshold,
            'suv_threshold': suv_threshold,
            'shape_threshold': shape_threshold,
            'overlap_threshold': overlap_threshold
        }
    }
    
    # Create statistics
    stats = PostProcessingStats(
        total_lesions=original_count,
        rule1_excluded=excluded_count_r1,
        rule2_symmetric_pairs=symmetric_pairs,
        rule3_excluded=excluded_count_r3,
        final_lesions=len(filtered_lesions_r3)
    )
    
    # Save processed data
    print(f"\nSaving processed data to: {output_json}")
    save_processed_data(output_data, output_json)
    
    # Generate and save summary report
    if summary_file:
        summary_report = generate_summary_report(data, output_data, stats)
        with open(summary_file, 'w') as f:
            f.write(summary_report)
        print(f"Summary report saved to: {summary_file}")
    
    # Print final summary
    print(f"\n=== POST-PROCESSING COMPLETE ===")
    print(f"Original lesions: {stats.total_lesions}")
    print(f"Excluded by Rule 1: {stats.rule1_excluded}")
    print(f"Excluded by Rule 3: {stats.rule3_excluded}")
    print(f"Symmetric pairs found: {stats.rule2_symmetric_pairs}")
    print(f"Final lesions: {stats.final_lesions}")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(description='Rule-based post-processing for PET-CT lesion analysis')
    parser.add_argument('--input_json', help='Input JSON file', type=str, default='lesion_results.json')
    parser.add_argument('--output_json', help='Output JSON file with post-processing results', type=str, default='lesion_results_postprocessed.json')
    parser.add_argument('--summary', help='Optional summary report file', default=None)
    parser.add_argument('--y-threshold', type=float, default=2.0,
                       help='Y coordinate difference threshold for symmetric lesion detection (default: 15.0)')
    parser.add_argument('--z-threshold', type=float, default=1.0,
                       help='Z coordinate difference threshold for symmetric lesion detection (default: 15.0)')
    parser.add_argument('--x-sum-threshold', type=float, default=3.0,
                       help='X coordinate sum threshold for bilateral symmetry |x1+x2| (default: 10.0)')
    parser.add_argument('--suv-threshold', type=float, default=1.0,
                       help='SUV difference threshold for symmetric lesion detection (default: 1.0)')
    parser.add_argument('--shape-threshold', type=float, default=0.3,
                       help='Shape similarity threshold for symmetric lesion detection (default: 0.3)')
    parser.add_argument('--overlap-threshold', type=float, default=30.0,
                       help='Organ overlap threshold for Rule 3 exclusion (default: 30.0)')
    
    args = parser.parse_args()
    
    # Call the core function
    output_data = run_rule_based_processing(
        input_json=args.input_json,
        output_json=args.output_json,
        y_threshold=args.y_threshold,
        z_threshold=args.z_threshold,
        x_sum_threshold=args.x_sum_threshold,
        suv_threshold=args.suv_threshold,
        shape_threshold=args.shape_threshold,
        overlap_threshold=args.overlap_threshold,
        summary_file=args.summary
    )


if __name__ == "__main__":
    main()
