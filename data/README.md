# Data included with the repository

Only public AutoPET-derived lesion-description training data and non-patient label mappings are included.

`fig1.pdf` is the publication-quality workflow figure used by the root README; `fig1.png` is its web-renderable preview.

```text
autopet_nnunet/train/     2,066 candidates proposed by HS-UNet
autopet_threshold/train/  5,855 candidates proposed by SUV thresholding
```

Both directories are serialized Hugging Face `Dataset` objects with three string fields:

- `input_text`: structured PET/CT candidate description;
- `output_text`: normalized `lesion_site` or `physiological_site` training target;
- `original_output`: target before normalization.

Load either dataset with:

```python
from datasets import load_from_disk

hs_unet = load_from_disk("data/autopet_nnunet/train")
threshold = load_from_disk("data/autopet_threshold/train")
```

Institutional images, reports, annotations, and report-conditioned training examples are not included. Obtain the underlying AutoPET images from the original data provider and follow its terms of use when redistributing derived data.

The corresponding trained LoRA adapters are distributed in [Zenodo record 20785543](https://zenodo.org/records/20785543). `autopet_nnunet/train` corresponds to the HS-UNet-trained adapter, while `autopet_threshold/train` corresponds to the SUV-threshold-trained adapter.
