# Inference and evaluation

All lesion-level inference entry points live in this directory:

```text
infer_gpt_oss.py              gpt-oss base model or GRPO adapter
infer_medgemma.py             MedGemma baseline
infer_api_models.py           Gemini and OpenAI API baselines
evaluate_predictions.py       classification and filtered-mask metrics
evaluate_segmentation_masks.py image-only segmentation metrics
```

`infer_gpt_oss.py` handles both report-free and report-conditioned inference; pass `--use_report` only when the input JSON contains a de-identified report under `metadata.pet_report`.

## AutoPET LoRA adapters

[Zenodo record 20785543](https://zenodo.org/records/20785543) contains two report-free LoRA adapters with different candidate-training distributions:

| Adapter | Use with |
|---|---|
| HS-UNet-trained AutoPET adapter | JSON descriptions and masks produced from HS-UNet candidates |
| SUV-threshold-trained AutoPET adapter | JSON descriptions and masks produced from SUV-threshold candidates |

Do not select an adapter solely by downstream dataset name: select it by the upstream candidate-generation method. For normal evaluation, the HS-UNet adapter should receive HS-UNet candidates and the threshold adapter should receive threshold candidates.

```bash
# HS-UNet candidates
python eval/infer_gpt_oss.py \
  --data_dir /path/to/hs_unet_descriptions \
  --model_path /path/to/hs_unet_trained_lora_adapter

# SUV-threshold candidates
python eval/infer_gpt_oss.py \
  --data_dir /path/to/threshold_descriptions \
  --model_path /path/to/threshold_trained_lora_adapter
```

Each extracted adapter directory should be kept outside Git and supplied directly via `--model_path`.

## Released evaluation results

The metric CSVs used for the manuscript are included in `evaluation_results/`:

- `evaluation_results/autopet/` contains AutoPET classification and segmentation results.
- `evaluation_results/osu/` contains institutional-cohort classification and segmentation results.

Only numeric per-case metrics and study case identifiers are included. Patient images, reports, prediction masks, and other raw institutional data are not released.
