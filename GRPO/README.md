# GRPO training

This directory contains the Group Relative Policy Optimization (GRPO) training code used to adapt `gpt-oss-20b` for lesion-level PET/CT adjudication. Inference is consolidated under `eval/`.

## Training task

Each example contains a structured description of one candidate uptake region. The description includes SUV statistics, volume and shape features, organ overlap and proximity, vertebral level, and body-centered coordinates. The report-conditioned variant additionally receives the corresponding de-identified radiology report.

The model must reason about the candidate and end with exactly one classification and anatomical site:

```text
physiological_site: [site]
```

or:

```text
lesion_site: [site]
```

The site must come from `data/lymphoma_site_lists.json`.

## Reward design

The main reward combines binary lesion classification with anatomical localization:

| Reward term | Value | Purpose |
|---|---:|---|
| Correct `lesion_site` versus `physiological_site` class | +1.0 | Optimizes the final lesion-retention decision |
| Exact anatomical site | +2.0 | Encourages anatomically specific reasoning |
| Accepted equivalent site group | +1.5 | Gives partial credit for clinically related labels |
| Both class labels emitted in one answer | -1.0 total | Penalizes an ambiguous final decision |
| Missing SUV-based reasoning | -0.5 | Encourages the reasoning trace to reference uptake intensity |
| Non-canonical site, report-conditioned model only | -1.0 | Keeps predictions within the predefined site vocabulary |

A fully correct class and exact site receives 3 points before the reasoning penalty. Equivalent groups provide partial localization credit for pairs such as `bone`/`bone_marrow`, skeletal variants, salivary-gland variants, and related gastrointestinal sites.

The report-free reasoning check accepts references such as `SUV max`, `max SUV`, or `max_suv`. The report-conditioned version also accepts an explicit reference to evidence from the radiology report. This check does not add a positive bonus; it applies a 0.5-point penalty when neither form of evidence appears.

`no_cheating` is currently a reserved reward hook and returns zero for every completion. It is kept in the trainer configuration so a leakage or copying penalty can be added without changing the training interface.

## Reward ablations

Two experimental reward variants were evaluated during development:

- **Flipped reward:** reversing the intended reward direction produced poorer lesion-classification performance. This confirmed that the improvement was not explained by sampling or training alone and that the direction of the task reward matters.
- **No-location reward:** removing the anatomical-site component and optimizing only the binary lesion/physiological decision also produced poorer performance. Anatomical localization therefore acts as useful auxiliary supervision rather than an incidental output field.

These ablations are not the released configuration. The training scripts implement the combined class-plus-location reward described above.

## Included AutoPET datasets

The report-free script can train directly from either public AutoPET-derived dataset:

| Candidate source | Dataset path | Candidates |
|---|---|---:|
| HS-UNet | `data/autopet_nnunet/train` | 2,066 |
| SUV thresholding | `data/autopet_threshold/train` | 5,855 |

Each is a serialized Hugging Face `Dataset` containing `input_text`, `output_text`, and `original_output`. Institutional report-conditioned examples are not distributed.

## Training configuration

| Setting | Report-free | Report-conditioned |
|---|---:|---:|
| Base model | `unsloth/gpt-oss-20b-BF16` | same |
| LoRA rank / alpha | 8 / 16 | 8 / 16 |
| Per-device batch size | 4 | 2 |
| Gradient accumulation | 4 | 4 |
| Generations per prompt | 4 | 4 |
| Micro-steps | 1,200 | 1,200 |
| Learning rate | 5e-5 | 5e-5 |
| Generation temperature | 1.0 | 1.0 |
| Maximum sequence / prompt length | 4,096 / 2,048 | same |

LoRA is applied to the attention projections (`q`, `k`, `v`, and `o`) and MLP projections (`gate`, `up`, and `down`). The optimizer is 8-bit AdamW with linear scheduling, 0.1 warmup ratio, and 0.001 weight decay.

## Usage

Install the LLM dependencies and run commands from the repository root:

```bash
python -m pip install -r requirements-llm.txt
```

Train on HS-UNet candidates:

```bash
python GRPO/training/train_grpo_no_reports.py --candidate-source hs-unet
```

Train on SUV-threshold candidates:

```bash
python GRPO/training/train_grpo_no_reports.py --candidate-source threshold
```

Override the bundled data or output location with `--data-dir` and `--output-dir`.

The report-conditioned model requires a private dataset supplied explicitly:

```bash
python GRPO/training/train_grpo_with_reports.py \
  --data-dir /path/to/private_report_dataset \
  --output-dir outputs/grpo_with_reports
```

Generated checkpoints and adapters are excluded from Git. The two released AutoPET adapters are archived in [Zenodo record 20785543](https://zenodo.org/records/20785543): one was trained on HS-UNet candidates and one on SUV-threshold candidates. At inference time, pair each adapter with descriptions generated by the same candidate method. Use the scripts under `eval/` for inference and evaluation.
