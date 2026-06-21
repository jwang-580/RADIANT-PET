#!/usr/bin/env python3
"""Unified preprocessing for RADIANT-PET candidate descriptions.

This entry point implements the paper's shared preprocessing path for either
SUV-threshold or HS-UNet candidate masks:

1. discover paired PET/CT cases;
2. optionally create TotalSegmentator organ masks;
3. generate threshold candidates or load HS-UNet candidates;
4. split candidates with the SUV-aware watershed procedure; and
5. write one labeled mask and one structured JSON description per case.

Raw images, organ masks, and predictions are inputs/outputs and are deliberately
not version-controlled.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, requires, version
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parent
DEFAULT_MAPPING = ROOT / "data" / "totalsegmentator_index_mapping.json"
TOTAL_TASKS = {
    "total": "total",
    "head_glands": "head_glands_cavities",
    "hn_vessels": "headneck_bones_vessels",
    "hn_muscle": "headneck_muscles",
}


@dataclass(frozen=True)
class Case:
    case_id: str
    pet: Path
    ct: Path


def strip_nii(path: Path) -> str:
    name = path.name
    return name[:-7] if name.endswith(".nii.gz") else path.stem


def discover_cases(input_dir: Path, pet_pattern: str, selected: set[str]) -> list[Case]:
    cases: list[Case] = []
    for pet in sorted(input_dir.glob(pet_pattern)):
        stem = strip_nii(pet)
        if not stem.endswith("_0001"):
            continue
        case_id = stem[:-5]
        if selected and case_id not in selected:
            continue
        ct = pet.with_name(f"{case_id}_0000.nii.gz")
        if not ct.exists():
            raise FileNotFoundError(f"Missing CT pair for {pet.name}: {ct.name}")
        cases.append(Case(case_id, pet.resolve(), ct.resolve()))
    if not cases:
        wanted = ", ".join(sorted(selected)) if selected else pet_pattern
        raise FileNotFoundError(f"No PET/CT cases found in {input_dir} for {wanted}")
    return cases


def load_mask_types(mapping_file: Path) -> list[str]:
    with mapping_file.open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    return list(mapping)


def find_organ_masks(case: Case, organ_dir: Path, mask_types: Iterable[str]) -> dict[str, str]:
    masks: dict[str, str] = {}
    bases = [f"{case.case_id}_0000", case.case_id]
    if case.case_id.endswith("_400"):
        bases.append(case.case_id[:-4] + "_400_0000")
    else:
        bases.append(case.case_id + "_400_0000")

    for mask_type in mask_types:
        candidates = [
            organ_dir / f"{base}_{mask_type}{ext}"
            for base in dict.fromkeys(bases)
            for ext in (".nii.gz", ".nii")
        ]
        match = next((path for path in candidates if path.exists()), None)
        if match:
            masks[mask_type] = str(match.resolve())
    return masks


def check_totalsegmentator_dependencies() -> None:
    """Fail early when the installed Torch violates TotalSegmentator metadata."""
    try:
        torch_version = version("torch")
        total_requirements = requires("TotalSegmentator") or []
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "TotalSegmentator and Torch must be installed in the active environment. "
            "Install requirements-preprocessing.txt in a dedicated preprocessing environment."
        ) from exc

    for raw_requirement in total_requirements:
        requirement = Requirement(raw_requirement)
        if canonicalize_name(requirement.name) != "torch":
            continue
        if requirement.marker and not requirement.marker.evaluate():
            continue
        if requirement.specifier and not requirement.specifier.contains(
            torch_version, prereleases=True
        ):
            raise RuntimeError(
                "TotalSegmentator dependency conflict: installed "
                f"torch=={torch_version}, but TotalSegmentator requires "
                f"torch{requirement.specifier}. Create a separate preprocessing "
                "environment and install requirements-preprocessing.txt; do not "
                "install the LLM/Unsloth requirements in that environment."
            )


def run_totalsegmentator(case: Case, organ_dir: Path, mask_types: Iterable[str], device: str) -> None:
    executable = shutil.which("TotalSegmentator")
    if not executable:
        raise RuntimeError(
            "TotalSegmentator is not installed or is not on PATH. Install the optional "
            "preprocessing dependency, or provide precomputed organ masks."
        )
    check_totalsegmentator_dependencies()
    organ_dir.mkdir(parents=True, exist_ok=True)
    for mask_type in mask_types:
        task = TOTAL_TASKS.get(mask_type)
        if task is None:
            raise ValueError(f"No TotalSegmentator task is configured for mapping group {mask_type!r}")
        output = organ_dir / f"{case.case_id}_0000_{mask_type}.nii.gz"
        if output.exists():
            continue
        command = [
            executable,
            "-i", str(case.ct),
            "-o", str(output),
            "--task", task,
            "--ml",
            "--device", device,
        ]
        subprocess.run(command, check=True)


def find_candidate_mask(case_id: str, mask_dir: Path) -> Path:
    names = (
        f"{case_id}.nii.gz",
        f"{case_id}_mask.nii.gz",
        f"{case_id}_t8_mask.nii.gz",
        f"{case_id}_400_t8_mask.nii.gz",
    )
    match = next((mask_dir / name for name in names if (mask_dir / name).exists()), None)
    if match is None:
        raise FileNotFoundError(
            f"No HS-UNet mask found for {case_id} in {mask_dir}; tried: {', '.join(names)}"
        )
    return match.resolve()


def describe_mask(
    lesion_mask: Path,
    case: Case,
    organ_masks: dict[str, str],
    mapping_file: Path,
    output_json: Path,
) -> None:
    # Reuse the feature implementation used by the original experiment code.
    from nnunet_processing.describe_candidates import process_single_case

    prefix = f"{case.case_id}_"
    args = SimpleNamespace(
        organ_mask_prefix=prefix,
        organ_mask_dir=str(next(iter(Path(p).parent for p in organ_masks.values()))),
        mapping_file=str(mapping_file),
        y_threshold=2.0,
        z_threshold=1.0,
        x_sum_threshold=3.0,
        suv_threshold=1.0,
        shape_threshold=0.3,
    )
    ok = process_single_case(
        str(lesion_mask),
        str(case.pet),
        str(output_json),
        args,
        current_organ_prefix=prefix,
        organ_mask_files_override=organ_masks,
    )
    if not ok:
        raise RuntimeError(f"Failed to generate lesion descriptions for {case.case_id}")


def preprocess_threshold(
    case: Case,
    organ_masks: dict[str, str],
    mapping_file: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    from suv_threshold_processing.threshold_candidates import run_suv_extraction

    mask_path = output_dir / f"{case.case_id}_candidates.nii.gz"
    json_path = output_dir / f"{case.case_id}_description.json"
    run_suv_extraction(
        suv_file=str(case.pet),
        organ_mask_files=organ_masks,
        mapping_file=str(mapping_file),
        output_json=str(json_path),
        save_mask_path=str(mask_path),
        suv_threshold=args.seed_suv,
        grow_suv_threshold=args.grow_suv,
        min_voxels=args.min_voxels,
        connectivity=args.connectivity,
        pre_min_voxels=args.min_voxels,
        post_min_voxels=args.min_voxels,
        exclude_brain=False,
        exclude_kidneys=False,
        exclude_bladder=False,
        z_boundary_voxels=args.z_boundary,
    )


def preprocess_hs_unet(
    case: Case,
    organ_masks: dict[str, str],
    mapping_file: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    from nnunet_processing.postprocess_hs_unet import process_lesion_mask_watershed

    raw_mask = find_candidate_mask(case.case_id, args.candidate_mask_dir)
    mask_path = output_dir / f"{case.case_id}_candidates.nii.gz"
    json_path = output_dir / f"{case.case_id}_description.json"
    process_lesion_mask_watershed(
        suv_file=str(case.pet),
        mask_file=str(raw_mask),
        output_file=str(mask_path),
        threshold=args.grow_suv,
        z_boundary=args.z_boundary,
        organ_file=None,
        organ_mapping=None,
    )
    describe_mask(mask_path, case, organ_masks, mapping_file, json_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create RADIANT-PET candidate masks and structured lesion descriptions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with <case>_0000 CT and <case>_0001 SUV PET NIfTI files")
    parser.add_argument(
        "--organ-mask-dir",
        type=Path,
        help=(
            "Directory containing precomputed TotalSegmentator masks. With "
            "--run-totalsegmentator, this is an optional destination and defaults "
            "to <output-dir>/organ_masks"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination for candidate masks and JSON descriptions")
    parser.add_argument("--source", choices=("threshold", "hs-unet"), default="threshold", help="Candidate-generation method")
    parser.add_argument("--candidate-mask-dir", type=Path, help="HS-UNet prediction directory (required for --source hs-unet)")
    parser.add_argument("--mapping-file", type=Path, default=DEFAULT_MAPPING, help="TotalSegmentator label mapping JSON")
    parser.add_argument("--pet-pattern", default="*_0001.nii.gz", help="PET file glob")
    parser.add_argument("--case-id", action="append", default=[], help="Only process this case ID; may be repeated")
    parser.add_argument("--run-totalsegmentator", action="store_true", help="Generate missing organ masks from CT")
    parser.add_argument("--device", default="gpu", help="TotalSegmentator device, for example gpu, gpu:0, or cpu")
    parser.add_argument("--seed-suv", type=float, default=3.5, help="Threshold candidate seed SUV")
    parser.add_argument("--grow-suv", type=float, default=2.5, help="HS-UNet mask SUV cutoff; threshold candidates also grow at SUV 2.5 in the core implementation")
    parser.add_argument("--min-voxels", type=int, default=20, help="Minimum candidate size")
    parser.add_argument("--connectivity", type=int, choices=(6, 18, 26), default=18)
    parser.add_argument("--z-boundary", type=int, default=2, help="Slices excluded at each axial boundary")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing per-case outputs")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed case")
    parser.add_argument("--dry-run", action="store_true", help="Validate discovery and print work without processing images")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.mapping_file = args.mapping_file.resolve()
    if args.organ_mask_dir is None:
        if not args.run_totalsegmentator:
            raise ValueError(
                "--organ-mask-dir is required unless --run-totalsegmentator is used"
            )
        args.organ_mask_dir = args.output_dir / "organ_masks"
    else:
        args.organ_mask_dir = args.organ_mask_dir.resolve()
    if not args.input_dir.is_dir():
        raise NotADirectoryError(args.input_dir)
    if not args.mapping_file.is_file():
        raise FileNotFoundError(args.mapping_file)
    if not args.run_totalsegmentator and not args.organ_mask_dir.is_dir():
        raise NotADirectoryError(args.organ_mask_dir)
    if args.source == "hs-unet":
        if args.candidate_mask_dir is None:
            raise ValueError("--candidate-mask-dir is required with --source hs-unet")
        args.candidate_mask_dir = args.candidate_mask_dir.resolve()
        if not args.candidate_mask_dir.is_dir():
            raise NotADirectoryError(args.candidate_mask_dir)
    if args.seed_suv <= args.grow_suv:
        raise ValueError("--seed-suv must be greater than --grow-suv")
    if args.min_voxels < 1 or args.z_boundary < 0:
        raise ValueError("--min-voxels must be positive and --z-boundary non-negative")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        mask_types = load_mask_types(args.mapping_file)
        cases = discover_cases(args.input_dir, args.pet_pattern, set(args.case_id))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Discovered {len(cases)} case(s); source={args.source}")
    failures: list[tuple[str, str]] = []

    for case in cases:
        mask_out = args.output_dir / f"{case.case_id}_candidates.nii.gz"
        json_out = args.output_dir / f"{case.case_id}_description.json"
        if not args.overwrite and mask_out.exists() and json_out.exists():
            print(f"[skip] {case.case_id}: outputs already exist")
            continue
        print(f"[case] {case.case_id}")
        if args.dry_run:
            continue
        try:
            organ_masks = find_organ_masks(case, args.organ_mask_dir, mask_types)
            if len(organ_masks) != len(mask_types) and args.run_totalsegmentator:
                run_totalsegmentator(case, args.organ_mask_dir, mask_types, args.device)
                organ_masks = find_organ_masks(case, args.organ_mask_dir, mask_types)
            missing = sorted(set(mask_types) - set(organ_masks))
            if missing:
                raise FileNotFoundError(
                    f"Missing organ masks for {case.case_id}: {', '.join(missing)}. "
                    "Use --run-totalsegmentator or provide precomputed masks."
                )
            if args.source == "threshold":
                preprocess_threshold(case, organ_masks, args.mapping_file, args.output_dir, args)
            else:
                preprocess_hs_unet(case, organ_masks, args.mapping_file, args.output_dir, args)
        except Exception as exc:  # keep batch runs useful while retaining a non-zero exit
            failures.append((case.case_id, str(exc)))
            print(f"[error] {case.case_id}: {exc}", file=sys.stderr)
            if args.fail_fast:
                break

    if failures:
        print("\nFailed cases:", file=sys.stderr)
        for case_id, message in failures:
            print(f"  {case_id}: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
