# Evaluation results

This directory contains the per-case metrics used for the RADIANT-PET manuscript.

- `autopet/`: AutoPET classification and segmentation metrics.
- `osu/`: institutional-cohort classification and segmentation metrics.

Classification files report candidate-level TP, TN, FP, FN, sensitivity, specificity, precision, F1, and accuracy by case. Segmentation files report Dice, sensitivity, specificity, precision, IoU, predicted volume, ground-truth volume, and intersection volume by case.

The release contains numeric metrics and study case identifiers only. It does not contain patient images, clinical reports, prediction masks, or other raw institutional data.
