#!/usr/bin/env python3
"""
Comprehensive evaluation script for PET-CT lesion analysis.

This script performs three main tasks:
1. Calculate classification metrics from JSON predictions (TP/TN/FP/FN, F1, sensitivity, specificity)
2. Filter segmentation files to keep only lesions classified as "lesion_site"
3. Calculate segmentation metrics (Dice, sensitivity, specificity) against ground truth

Usage:
    python eval/evaluate_predictions.py \
        --json_dir results/lora_report_1200_results \
        --seg_dir nnunet/nnunet_raw/dataset002_test/results \
        --gt_dir nnunet/nnunet_raw/dataset002_test/gt \
        --output_dir eval/evaluation_results
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClassificationMetrics:
    """Calculate and store classification metrics."""
    
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
    
    def add(self, predicted_type: str, is_true_lesion: bool):
        """Add a single prediction to the metrics."""
        is_predicted_lesion = (predicted_type == "lesion_site")
        
        if is_predicted_lesion and is_true_lesion:
            self.tp += 1
        elif not is_predicted_lesion and not is_true_lesion:
            self.tn += 1
        elif is_predicted_lesion and not is_true_lesion:
            self.fp += 1
        elif not is_predicted_lesion and is_true_lesion:
            self.fn += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """Calculate all classification metrics."""
        metrics = {
            'TP': self.tp,
            'TN': self.tn,
            'FP': self.fp,
            'FN': self.fn,
        }
        
        # Sensitivity (Recall)
        if (self.tp + self.fn) > 0:
            metrics['Sensitivity'] = self.tp / (self.tp + self.fn)
        else:
            metrics['Sensitivity'] = 0.0
        
        # Specificity
        if (self.tn + self.fp) > 0:
            metrics['Specificity'] = self.tn / (self.tn + self.fp)
        else:
            metrics['Specificity'] = 0.0
        
        # Precision
        if (self.tp + self.fp) > 0:
            metrics['Precision'] = self.tp / (self.tp + self.fp)
        else:
            metrics['Precision'] = 0.0
        
        # F1 Score
        if metrics['Precision'] + metrics['Sensitivity'] > 0:
            metrics['F1'] = 2 * (metrics['Precision'] * metrics['Sensitivity']) / \
                           (metrics['Precision'] + metrics['Sensitivity'])
        else:
            metrics['F1'] = 0.0
        
        # Accuracy
        total = self.tp + self.tn + self.fp + self.fn
        if total > 0:
            metrics['Accuracy'] = (self.tp + self.tn) / total
        else:
            metrics['Accuracy'] = 0.0
        
        return metrics


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
        
        # Calculate true negatives
        tn = np.logical_and(~pred_bool, ~gt_bool).sum()
        fp = np.logical_and(pred_bool, ~gt_bool).sum()
        
        metrics = {}
        
        # Dice Coefficient
        if (pred_sum + gt_sum) > 0:
            metrics['Dice'] = 2 * intersection / (pred_sum + gt_sum)
        else:
            metrics['Dice'] = 0.0 if union == 0 else 0.0
        
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


def process_json_file(json_path: Path) -> Tuple[ClassificationMetrics, List[int], str]:
    """
    Process a single JSON file to extract classification metrics and lesion IDs to keep.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Tuple of (ClassificationMetrics, list of lesion IDs to keep, case_id)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    metrics = ClassificationMetrics()
    lesion_ids_to_keep = []
    
    for lesion in data.get('lesions', []):
        lesion_id = lesion.get('id')
        predicted_type = lesion.get('predicted_type', '')
        is_true_lesion = (
            lesion.get("is_true_lesion")
            if "is_true_lesion" in lesion
            else lesion.get("ground_truth_validation", {}).get("is_true_positive", False)
        )

        
        # Add to classification metrics
        metrics.add(predicted_type, is_true_lesion)
        
        # Keep lesion if predicted as lesion_site
        if predicted_type == "lesion_site":
            lesion_ids_to_keep.append(lesion_id)
    
    # Extract case_id from filename (e.g., OSU02_acc0_description.json -> OSU02_acc0)
    case_id = json_path.stem.replace('_description', '').replace('_results', '')
    
    return metrics, lesion_ids_to_keep, case_id


def filter_segmentation(seg_path: Path, lesion_ids_to_keep: List[int], 
                       output_path: Path) -> None:
    """
    Filter segmentation file to keep only specified lesion IDs.
    
    Args:
        seg_path: Path to original segmentation file
        lesion_ids_to_keep: List of lesion IDs to retain
        output_path: Path to save filtered segmentation
    """
    # Load segmentation
    seg_nii = nib.load(seg_path)
    seg_data = seg_nii.get_fdata().astype(int)  # Convert to int to handle float IDs
    
    # Create filtered mask
    filtered_data = np.zeros_like(seg_data)
    for lesion_id in lesion_ids_to_keep:
        filtered_data[seg_data == lesion_id] = lesion_id
    
    # Save filtered segmentation
    filtered_nii = nib.Nifti1Image(filtered_data, seg_nii.affine, seg_nii.header)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(filtered_nii, output_path)
    
    logger.info(f"Filtered segmentation saved to {output_path}")


def calculate_classification_metrics(json_dir: Path, output_dir: Path) -> pd.DataFrame:
    """
    Calculate classification metrics for all JSON files.
    
    Args:
        json_dir: Directory containing JSON files
        output_dir: Directory to save results
        
    Returns:
        DataFrame with per-file and aggregate metrics
    """
    logger.info("Step 1: Calculating classification metrics...")
    
    json_files = sorted(json_dir.glob("*.json"))
    results = []
    aggregate_metrics = ClassificationMetrics()
    
    for json_path in tqdm(json_files, desc="Processing JSON files"):
        metrics, _, case_id = process_json_file(json_path)
        
        # Add to aggregate
        aggregate_metrics.tp += metrics.tp
        aggregate_metrics.tn += metrics.tn
        aggregate_metrics.fp += metrics.fp
        aggregate_metrics.fn += metrics.fn
        
        # Store per-file results
        file_metrics = metrics.get_metrics()
        file_metrics['Case_ID'] = case_id
        results.append(file_metrics)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['Case_ID', 'TP', 'TN', 'FP', 'FN', 'Sensitivity', 'Specificity', 
            'Precision', 'F1', 'Accuracy']
    df = df[cols]
    
    # Add aggregate row
    agg_metrics = aggregate_metrics.get_metrics()
    agg_metrics['Case_ID'] = 'AGGREGATE'
    df = pd.concat([df, pd.DataFrame([agg_metrics])], ignore_index=True)
    
    # Save to CSV
    output_path = output_dir / "classification_metrics.csv"
    df.to_csv(output_path, index=False, float_format='%.4f')
    logger.info(f"Classification metrics saved to {output_path}")
    
    return df


def filter_all_segmentations(json_dir: Path, seg_dir: Path, 
                             filtered_seg_dir: Path) -> Dict[str, List[int]]:
    """
    Filter all segmentation files based on JSON predictions.
    
    Args:
        json_dir: Directory containing JSON files
        seg_dir: Directory containing original segmentation files
        filtered_seg_dir: Directory to save filtered segmentations
        
    Returns:
        Dictionary mapping case_id to list of kept lesion IDs
    """
    logger.info("Step 2: Filtering segmentation files...")
    
    json_files = sorted(json_dir.glob("*.json"))
    kept_lesions = {}
    
    for json_path in tqdm(json_files, desc="Filtering segmentations"):
        _, lesion_ids_to_keep, case_id = process_json_file(json_path)
        kept_lesions[case_id] = lesion_ids_to_keep
        
        # Find corresponding segmentation file (check both .nii.gz and .nii)
        seg_files = list(seg_dir.glob(f"{case_id}_*.nii.gz"))
        if not seg_files:
            seg_files = list(seg_dir.glob(f"{case_id}_*.nii"))
        if not seg_files:
            logger.warning(f"No segmentation file found for {case_id}")
            continue
        
        seg_path = seg_files[0]
        output_path = filtered_seg_dir / f"{case_id}_filtered.nii.gz"
        
        filter_segmentation(seg_path, lesion_ids_to_keep, output_path)
    
    return kept_lesions


def calculate_segmentation_metrics(filtered_seg_dir: Path, gt_dir: Path, 
                                   output_dir: Path) -> pd.DataFrame:
    """
    Calculate segmentation metrics comparing filtered predictions with ground truth.
    
    Args:
        filtered_seg_dir: Directory containing filtered segmentation files
        gt_dir: Directory containing ground truth segmentations
        output_dir: Directory to save results
        
    Returns:
        DataFrame with per-file and aggregate metrics
    """
    logger.info("Step 3: Calculating segmentation metrics...")
    
    filtered_files = sorted(filtered_seg_dir.glob("*_filtered.nii.gz"))
    results = []
    
    # Aggregate metrics
    total_dice = []
    total_sensitivity = []
    total_specificity = []
    total_precision = []
    total_iou = []
    
    for filtered_path in tqdm(filtered_files, desc="Calculating segmentation metrics"):
        # Extract case_id (remove _filtered.nii.gz to get OSU02_acc0)
        # filtered_path.name is like "OSU02_acc0_filtered.nii.gz"
        case_id = filtered_path.name.replace('_filtered.nii.gz', '')
        
        # Find ground truth file
        gt_path = gt_dir / f"{case_id}_gt.nii.gz"
        if not gt_path.exists():
            gt_path = gt_dir / f"{case_id}_gt_dilate.nii.gz"
        if not gt_path.exists():
            logger.warning(f"Ground truth not found for {case_id}: {gt_path}")
            continue
        
        # Load images
        pred_nii = nib.load(filtered_path)
        gt_nii = nib.load(gt_path)
        
        pred_data = pred_nii.get_fdata()
        gt_data = gt_nii.get_fdata()
        
        # Calculate metrics
        metrics = SegmentationMetrics.calculate_metrics(pred_data, gt_data)
        metrics['Case_ID'] = case_id
        results.append(metrics)
        
        # Collect for aggregate
        total_dice.append(metrics['Dice'])
        total_sensitivity.append(metrics['Sensitivity'])
        total_specificity.append(metrics['Specificity'])
        total_precision.append(metrics['Precision'])
        total_iou.append(metrics['IoU'])
    
    # Create DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # Reorder columns
        cols = ['Case_ID', 'Dice', 'Sensitivity', 'Specificity', 'Precision', 'IoU',
                'Pred_Volume', 'GT_Volume', 'Intersection_Volume']
        df = df[cols]
        
        # Add aggregate row (mean across all cases)
        agg_metrics = {
            'Case_ID': 'AGGREGATE_MEAN',
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
        logger.warning("No ground truth files found. Creating empty metrics file.")
        df = pd.DataFrame(columns=['Case_ID', 'Dice', 'Sensitivity', 'Specificity', 
                                   'Precision', 'IoU', 'Pred_Volume', 'GT_Volume', 
                                   'Intersection_Volume'])
    
    # Save to CSV
    output_path = output_dir / "segmentation_metrics.csv"
    df.to_csv(output_path, index=False, float_format='%.4f')
    logger.info(f"Segmentation metrics saved to {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of PET-CT lesion predictions"
    )
    parser.add_argument(
        '--json_dir',
        type=str,
        default='results/autopet/lora_nnunet_autopet_trained_eval_nnunet_autopet_results',
        help='Directory containing JSON prediction files'
    )
    parser.add_argument(
        '--seg_dir',
        type=str,
        default='nnunet/nnunet_raw/dataset004_autopet_test/labelstr/nnunet_t8',
        help='Directory containing original segmentation files'
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
        default='eval/evaluation_results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--filtered_seg_dir',
        type=str,
        default='nnunet/nnunet_raw/dataset004_autopet_test/filtered_results',
        help='Directory to save filtered segmentation files'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    json_dir = Path(args.json_dir)
    seg_dir = Path(args.seg_dir)
    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir)
    filtered_seg_dir = Path(args.filtered_seg_dir)
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_seg_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate input directories
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")
    if not seg_dir.exists():
        raise FileNotFoundError(f"Segmentation directory not found: {seg_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")
    
    logger.info("="*80)
    logger.info("Starting comprehensive evaluation")
    logger.info("="*80)
    logger.info(f"JSON directory: {json_dir}")
    logger.info(f"Segmentation directory: {seg_dir}")
    logger.info(f"Ground truth directory: {gt_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Filtered segmentation directory: {filtered_seg_dir}")
    logger.info("="*80)
    
    # Step 1: Calculate classification metrics
    classification_df = calculate_classification_metrics(json_dir, output_dir)
    logger.info("\nClassification Metrics Summary:")
    logger.info(f"\n{classification_df.tail(1).to_string(index=False)}")
    
    # Step 2: Filter segmentations
    kept_lesions = filter_all_segmentations(json_dir, seg_dir, filtered_seg_dir)
    total_kept = sum(len(ids) for ids in kept_lesions.values())
    logger.info(f"\nFiltered {len(kept_lesions)} segmentation files")
    logger.info(f"Total lesions kept: {total_kept}")
    
    # Step 3: Calculate segmentation metrics
    segmentation_df = calculate_segmentation_metrics(filtered_seg_dir, gt_dir, output_dir)
    logger.info("\nSegmentation Metrics Summary:")
    logger.info(f"\n{segmentation_df.tail(1).to_string(index=False)}")
    
    logger.info("="*80)
    logger.info("Evaluation complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
