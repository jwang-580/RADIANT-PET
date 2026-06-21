#!/usr/bin/env python3
"""Run HS-UNet inference on paired CT/PET NIfTI cases."""

from __future__ import annotations

import argparse
import gc
import os
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def configure_nnunet_paths(input_dir: Path, model_dir: Path) -> None:
    """Set conventional nnU-Net paths when the checkout layout makes them clear."""
    raw_dir = next(
        (parent for parent in (input_dir, *input_dir.parents) if parent.name == "nnUNet_raw"),
        None,
    )
    nnunet_root = raw_dir.parent if raw_dir is not None else model_dir.parent
    os.environ.setdefault("nnUNet_raw", str(raw_dir or nnunet_root / "nnUNet_raw"))
    os.environ.setdefault("nnUNet_preprocessed", str(nnunet_root / "nnUNet_preprocessed"))
    os.environ.setdefault("nnUNet_results", str(nnunet_root / "nnUNet_results"))


def configure_checkpoint_loading(torch_module, trust_checkpoints: bool) -> None:
    """Handle the weights_only default introduced by PyTorch 2.6."""
    match = re.match(r"(\d+)\.(\d+)", torch_module.__version__)
    torch_version = tuple(map(int, match.groups())) if match else (0, 0)
    if torch_version < (2, 6):
        return
    if not trust_checkpoints:
        raise RuntimeError(
            f"PyTorch {torch_module.__version__} defaults to safe weights-only loading, "
            "but this nnU-Net checkpoint contains additional serialized training state. "
            "If these model files are from a source you trust, rerun with "
            "--trust-checkpoints. Otherwise, do not load them. Alternatively, use the "
            "pinned preprocessing environment with PyTorch <2.6."
        )
    # nnUNetv2 calls torch.load without a weights_only argument. This documented
    # PyTorch compatibility switch restores the pre-2.6 behavior for that call.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    print(
        "Trusted-checkpoint mode enabled: allowing legacy pickle loading for "
        f"nnU-Net checkpoints under PyTorch {torch_module.__version__}."
    )


def load_predictor_class():
    """Load nnUNetPredictor from the environment or the original local source tree."""
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        return nnUNetPredictor
    except ModuleNotFoundError as installed_error:
        try:
            installed_version = version("nnunetv2")
        except PackageNotFoundError:
            installed_version = None

        local_source = (
            Path(__file__).resolve().parents[1]
            / "nnunet"
            / "autopet_3_submission"
        )
        if (local_source / "nnunetv2").is_dir():
            sys.path.insert(0, str(local_source))
            try:
                from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

                print(f"Using local nnUNetv2 source: {local_source}")
                return nnUNetPredictor
            except ModuleNotFoundError as local_error:
                raise ModuleNotFoundError(
                    f"Found the local nnUNetv2 source at {local_source}, but "
                    f"dependency {local_error.name!r} is missing from "
                    f"{sys.executable}. Install the local source and all of its "
                    "dependencies with:\n"
                    f"  {sys.executable} -m pip install -e {local_source}\n"
                    "Then rerun the import verification command."
                ) from local_error

        if installed_version is not None:
            raise ModuleNotFoundError(
                f"Package metadata reports nnunetv2=={installed_version} in the "
                f"environment for {sys.executable}, but the `nnunetv2` module "
                "cannot be imported. This is usually a stale editable install or "
                "an incomplete package directory. Repair it with:\n"
                f"  {sys.executable} -m pip uninstall -y nnunetv2\n"
                f"  {sys.executable} -m pip install --no-cache-dir --no-deps "
                f"'nnunetv2=={installed_version}'\n"
                "Then verify with:\n"
                f"  {sys.executable} -c \"import nnunetv2; "
                "print(nnunetv2.__file__)\""
            ) from installed_error

        raise ModuleNotFoundError(
            "nnunetv2 is not importable by the Python interpreter running this "
            f"script ({sys.executable}). Install it with:\n"
            f"  {sys.executable} -m pip install 'nnunetv2>=2.5'\n"
            "Then verify with:\n"
            f"  {sys.executable} -c \"import nnunetv2; print(nnunetv2.__file__)\"\n"
            "Using a standalone `pip` command may install into a different "
            "Windows, WSL, Conda, or virtual-environment interpreter."
        ) from installed_error


def configure_autopet_trainer() -> bool:
    """Use the official AutoPET trainer, or install a stock-nnU-Net fallback.

    Returns True when the inference-only fallback is active.
    """
    import nnunetv2
    import nnunetv2.inference.predict_from_raw_data as prediction_module

    trainer_root = str(
        Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer"
    )
    trainer = prediction_module.recursive_find_python_class(
        trainer_root,
        "autoPET3_Trainer",
        "nnunetv2.training.nnUNetTrainer",
    )
    if trainer is not None:
        print(f"Using AutoPET trainer: {trainer.__module__}.{trainer.__name__}")
        return False

    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    original_resolver = prediction_module.recursive_find_python_class

    class autoPET3_Trainer(nnUNetTrainer):
        pass

    def resolve_trainer(folder, class_name, current_module):
        if class_name == "autoPET3_Trainer":
            return autoPET3_Trainer
        return original_resolver(folder, class_name, current_module)

    prediction_module.recursive_find_python_class = resolve_trainer
    print("Using stock nnU-Net inference compatibility for autoPET3_Trainer.")
    return True


def remove_fallback_only_parameters(predictor) -> None:
    """Remove AutoPET auxiliary heads only when using the stock-network fallback."""
    network_parameters = set(predictor.network.state_dict())
    allowed_prefix = "decoder.organ_seg_layers."
    removed = 0
    for fold, parameters in enumerate(predictor.list_of_parameters):
        unexpected = set(parameters) - network_parameters
        unsupported = sorted(
            key for key in unexpected if not key.startswith(allowed_prefix)
        )
        if unsupported:
            preview = ", ".join(unsupported[:5])
            raise RuntimeError(
                f"Fold {fold} contains unsupported checkpoint parameters: {preview}"
            )
        for key in unexpected:
            parameters.pop(key)
            removed += 1
    print(
        f"Removed {removed} auxiliary organ-head parameters for stock nnU-Net "
        "inference."
    )


def discover_inputs(input_dir: Path) -> tuple[list[list[str]], list[str]]:
    inputs: list[list[str]] = []
    case_ids: list[str] = []
    for ct in sorted(input_dir.glob("*_0000.nii.gz")):
        case_id = ct.name[:-12]
        pet = input_dir / f"{case_id}_0001.nii.gz"
        if not pet.exists():
            raise FileNotFoundError(f"Missing PET pair for {ct.name}: {pet.name}")
        inputs.append([str(ct.resolve()), str(pet.resolve())])
        case_ids.append(case_id)
    if not inputs:
        raise FileNotFoundError(f"No *_0000.nii.gz CT files found in {input_dir}")
    return inputs, case_ids


def predict(
    input_dir: Path,
    output_dir: Path,
    model_dir: Path,
    device_name: str,
    folds: list[int] | None,
    checkpoint: str,
    overwrite: bool,
    trust_checkpoints: bool,
) -> None:
    import torch

    configure_checkpoint_loading(torch, trust_checkpoints)
    configure_nnunet_paths(input_dir, model_dir)
    nnUNetPredictor = load_predictor_class()
    using_stock_fallback = configure_autopet_trainer()

    device = torch.device(device_name)
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

    input_files, case_ids = discover_inputs(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = [str(output_dir / f"{case_id}.nii.gz") for case_id in case_ids]

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=device.type == "cuda",
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_dir),
        use_folds=folds,
        checkpoint_name=checkpoint,
    )
    if using_stock_fallback:
        remove_fallback_only_parameters(predictor)
    predictor.predict_from_files(
        input_files,
        output_files,
        save_probabilities=False,
        overwrite=overwrite,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the RADIANT-PET HS-UNet over paired CT/PET cases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing <case>_0000 CT and <case>_0001 PET files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Prediction directory")
    parser.add_argument("--model-dir", type=Path, required=True, help="Extracted nnUNet trained-model directory")
    parser.add_argument("--device", default="cuda", help="PyTorch device such as cuda, cuda:0, or cpu")
    parser.add_argument("--folds", nargs="*", type=int, help="Folds to ensemble; omit to use all available folds")
    parser.add_argument("--checkpoint", default="checkpoint_best.pth")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--trust-checkpoints",
        action="store_true",
        help=(
            "Allow legacy pickle loading required by nnU-Net checkpoints under "
            "PyTorch >=2.6; use only for model files from a trusted source"
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        parser.error(f"Input directory not found: {input_dir}")
    if not model_dir.is_dir():
        parser.error(f"Model directory not found: {model_dir}")
    predict(
        input_dir,
        output_dir,
        model_dir,
        args.device,
        args.folds,
        args.checkpoint,
        args.overwrite,
        args.trust_checkpoints,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
