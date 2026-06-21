#!/usr/bin/env python3
"""
Evaluation script for PET-CT lesion segmentation masks.

This script evaluates predicted lesion masks directly against ground truth masks.
Unlike evaluate_predictions.py, this script does not involve JSON files and works
purely with NIfTI mask files (.nii or .nii.gz).

The script calculates segmentation metrics (Dice, sensitivity, specificity, etc.)
by comparing predicted masks with ground truth masks.

Usage:
    python eval/evaluate_segmentation_masks.py \
        --pred_dir nnunet/nnunet_raw/dataset002_test/predictions \
        --gt_dir nnunet/nnunet_raw/dataset002_test/gt \
        --output_dir eval/evaluation_results_mask
"""

import argparse
import os
from pathlib import Path
from typing import Dict
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm


class SegmentationMetrics:
    """Calculate segmentation quality metrics."""
    
    @staticmethod
    def calculate_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
        """
        Calculate segmentation metrics between prediction and ground truth.
        
        Args:
            pred: Binary prediction mask
            gt: Binary ground truth mask
            
        Returns:
            Dictionary of metrics
        """
        pred_bool = pred > 0
        gt_bool = gt > 0
        
        # Calculate intersection and union
        intersection = np.logical_and(pred_bool, gt_bool).sum()
        pred_sum = pred_bool.sum()
        gt_sum = gt_bool.sum()
        union = np.logical_or(pred_bool, gt_bool).sum()
        
        # Calculate true negatives and false positives
        tn = np.logical_and(~pred_bool, ~gt_bool).sum()
        fp = np.logical_and(pred_bool, ~gt_bool).sum()
        
        metrics = {}
        
        # Dice Coefficient
        if (pred_sum + gt_sum) > 0:
            metrics['Dice'] = 2 * intersection / (pred_sum + gt_sum)
        else:
            # Both masks are empty - perfect agreement
            metrics['Dice'] = 1.0
        
        # Sensitivity (Recall)
        if gt_sum > 0:
            metrics['Sensitivity'] = intersection / gt_sum
        else:
            metrics['Sensitivity'] = 1.0 if pred_sum == 0 else 0.0
        
        # Specificity
        if (tn + fp) > 0:
            metrics['Specificity'] = tn / (tn + fp)
        else:
            metrics['Specificity'] = 1.0
        
        # Precision
        if pred_sum > 0:
            metrics['Precision'] = intersection / pred_sum
        else:
            metrics['Precision'] = 1.0 if gt_sum == 0 else 0.0
        
        # IoU (Jaccard)
        if union > 0:
            metrics['IoU'] = intersection / union
        else:
            metrics['IoU'] = 1.0
        
        # Volume metrics
        metrics['Pred_Volume'] = int(pred_sum)
        metrics['GT_Volume'] = int(gt_sum)
        metrics['Intersection_Volume'] = int(intersection)
        
        return metrics


def find_matching_gt_file(pred_path: Path, gt_dir: Path) -> Path:
    """
    Find the matching ground truth file for a prediction file.
    
    Args:
        pred_path: Path to prediction file
        gt_dir: Directory containing ground truth files
        
    Returns:
        Path to matching ground truth file
        
    Raises:
        FileNotFoundError: If no matching ground truth file is found
    """
    # Extract case_id from prediction filename
    # Handle various naming patterns:
    # - OSU02_acc0_pred.nii.gz -> OSU02_acc0
    # - OSU02_acc0.nii.gz -> OSU02_acc0
    # - OSU02_acc0_400_0000.nii.gz -> OSU02_acc0
    
    filename = pred_path.name
    
    # Remove .nii.gz or .nii extension
    if filename.endswith('.nii.gz'):
        base_name = filename[:-7]
    elif filename.endswith('.nii'):
        base_name = filename[:-4]
    else:
        base_name = filename
    
    # Remove common suffixes
    for suffix in ['_pred', '_prediction', '_filtered', '_400_0000', '_t8_mask_watershed', '_mask']:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
    
    # Try to find matching ground truth file
    # Common patterns: case_id_gt.nii.gz, case_id.nii.gz
    possible_gt_names = [
        f"{base_name}_gt.nii.gz",
        f"{base_name}_gt.nii",
        f"{base_name}_gt_dilate.nii.gz",
        f"{base_name}.nii.gz",
        f"{base_name}.nii",
    ]
    
    for gt_name in possible_gt_names:
        gt_path = gt_dir / gt_name
        if gt_path.exists():
            return gt_path
    
    raise FileNotFoundError(f"No ground truth file found for {pred_path.name}")


def calculate_segmentation_metrics(pred_dir: Path, gt_dir: Path, 
                                   output_dir: Path) -> pd.DataFrame:
    """
    Calculate segmentation metrics comparing predictions with ground truth.
    
    Args:
        pred_dir: Directory containing predicted segmentation files
        gt_dir: Directory containing ground truth segmentations
        output_dir: Directory to save results
        
    Returns:
        DataFrame with per-file and aggregate metrics
    """
    print("Calculating segmentation metrics...")
    
    # Find all prediction files (.nii and .nii.gz)
    pred_files = sorted(list(pred_dir.glob("*.nii.gz")) + list(pred_dir.glob("*.nii")))
    
    if not pred_files:
        print(f"Warning: No prediction files found in {pred_dir}")
        return pd.DataFrame()
    
    results = []
    
    # Aggregate metrics
    total_dice = []
    total_sensitivity = []
    total_specificity = []
    total_precision = []
    total_iou = []
    
    for pred_path in tqdm(pred_files, desc="Calculating segmentation metrics"):
        try:
            # Find matching ground truth file
            gt_path = find_matching_gt_file(pred_path, gt_dir)
            
            # Load images
            pred_nii = nib.load(pred_path)
            gt_nii = nib.load(gt_path)
            
            pred_data = pred_nii.get_fdata()
            gt_data = gt_nii.get_fdata()
            
            # Check if shapes match
            if pred_data.shape != gt_data.shape:
                print(f"Warning: Shape mismatch for {pred_path.name}: "
                      f"pred {pred_data.shape} vs gt {gt_data.shape}")
                continue
            
            # Calculate metrics
            metrics = SegmentationMetrics.calculate_metrics(pred_data, gt_data)
            
            # Extract case_id for reporting
            if pred_path.name.endswith('.nii.gz'):
                case_id = pred_path.name[:-7]
            elif pred_path.name.endswith('.nii'):
                case_id = pred_path.name[:-4]
            else:
                case_id = pred_path.name
            
            metrics['Case_ID'] = case_id
            metrics['GT_File'] = gt_path.name
            results.append(metrics)
            
            # Collect for aggregate
            total_dice.append(metrics['Dice'])
            total_sensitivity.append(metrics['Sensitivity'])
            total_specificity.append(metrics['Specificity'])
            total_precision.append(metrics['Precision'])
            total_iou.append(metrics['IoU'])
            
        except FileNotFoundError as e:
            print(f"Warning: Skipping {pred_path.name}: {e}")
            continue
        except Exception as e:
            print(f"Error: Error processing {pred_path.name}: {e}")
            continue
    
    # Create DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # Reorder columns
        cols = ['Case_ID', 'GT_File', 'Dice', 'Sensitivity', 'Specificity', 
                'Precision', 'IoU', 'Pred_Volume', 'GT_Volume', 'Intersection_Volume']
        df = df[cols]
        
        # Add aggregate row (mean across all cases)
        # Note: This calculates mean Dice (average of per-case Dice scores)
        # which differs from overall Dice (2*sum(intersection)/sum(pred+gt))
        agg_metrics = {
            'Case_ID': 'AGGREGATE_MEAN',
            'GT_File': f'n={len(results)}',
            'Dice': np.mean(total_dice),
            'Sensitivity': np.mean(total_sensitivity),
            'Specificity': np.mean(total_specificity),
            'Precision': np.mean(total_precision),
            'IoU': np.mean(total_iou),
            'Pred_Volume': df['Pred_Volume'].sum(),
            'GT_Volume': df['GT_Volume'].sum(),
            'Intersection_Volume': df['Intersection_Volume'].sum()
        }
        df = pd.concat([df, pd.DataFrame([agg_metrics])], ignore_index=True)
    else:
        # Create empty DataFrame with proper columns if no results
        print("Warning: No valid prediction-ground truth pairs found. Creating empty metrics file.")
        df = pd.DataFrame(columns=['Case_ID', 'GT_File', 'Dice', 'Sensitivity', 
                                   'Specificity', 'Precision', 'IoU', 'Pred_Volume', 
                                   'GT_Volume', 'Intersection_Volume'])
    
    # Save to CSV
    output_path = output_dir / "segmentation_metrics.csv"
    df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"Segmentation metrics saved to {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate lesion segmentation masks against ground truth"
    )
    parser.add_argument(
        '--pred_dir',
        type=str,
        required=True,
        help='Directory containing predicted segmentation files (.nii or .nii.gz)'
    )
    parser.add_argument(
        '--gt_dir',
        type=str,
        default='nnunet/nnunet_raw/dataset004_autopet_test/gt',
        help='Directory containing ground truth segmentation files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='eval/evaluation_results_mask',
        help='Directory to save evaluation results'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate input directories
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")
    
    print("="*80)
    print("Starting segmentation mask evaluation")
    print("="*80)
    print(f"Prediction directory: {pred_dir}")
    print(f"Ground truth directory: {gt_dir}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Calculate segmentation metrics
    segmentation_df = calculate_segmentation_metrics(pred_dir, gt_dir, output_dir)
    
    if not segmentation_df.empty:
        print("\nSegmentation Metrics Summary:")
        print(f"\n{segmentation_df.tail(1).to_string(index=False)}")
    
    print("="*80)
    print("Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
