# Separate HS-UNet candidates with the shared watershed logic and an SUV 2.5 cutoff.

import argparse
import numpy as np
import nibabel as nib
import json
import os
import glob
import sys
from scipy import ndimage as ndi
from typing import Tuple

# Add parent directory to path to import from utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.lesion_features import (
    split_component_watershed, 
    bbox_from_mask, 
    connectivity_structure, 
    final_ccl_from_labelmap,
    create_exclusion_mask,
    labels_by_contains
)
from utils.mask_expansion import expand_mask_array

def load_organ_mapping(mapping_file):
    """Load organ mapping from JSON file"""
    if not os.path.exists(mapping_file):
        print(f"Warning: Mapping file not found: {mapping_file}")
        return {}
        
    with open(mapping_file, 'r') as f:
        data = json.load(f)
        
    # If "total" key exists (TotalSegmentator format), use it
    if 'total' in data:
         return {int(k): v for k, v in data['total'].items()}
    
    # Fallback: try to flatten all keys if they are groups
    mapping = {}
    for k, v in data.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                mapping[int(sub_k)] = sub_v
        else:
             try:
                mapping[int(k)] = v
             except ValueError:
                pass
    return mapping

def create_z_boundary_exclusion_mask(image_shape: Tuple[int, int, int], 
                                    boundary_voxels: int = 2) -> np.ndarray:
    """
    Create a mask to exclude boundary regions along the Z-axis (head-to-foot direction)
    to remove edge artifacts and noise commonly found at scan boundaries
    """
    z_boundary_mask = np.zeros(image_shape, dtype=bool)
    z_size = image_shape[2]
    
    if boundary_voxels > 0 and z_size > 2 * boundary_voxels:
        z_boundary_mask[:, :, :boundary_voxels] = True
        z_boundary_mask[:, :, -boundary_voxels:] = True
    
    return z_boundary_mask

def resample_mask_to_reference(mask_data, mask_affine, ref_affine, ref_shape):
    """
    Resample mask to match reference image voxel grid.
    """
    print("  Resampling mask to match reference affine...")
    i, j, k = np.mgrid[0:ref_shape[0], 0:ref_shape[1], 0:ref_shape[2]]
    ref_voxel_coords = np.column_stack([i.ravel(), j.ravel(), k.ravel(), np.ones(i.size)])
    ras_coords = ref_affine @ ref_voxel_coords.T
    mask_affine_inv = np.linalg.inv(mask_affine)
    mask_voxel_coords = mask_affine_inv @ ras_coords
    mask_coords = mask_voxel_coords[:3, :].T
    mask_coords = mask_coords.reshape(ref_shape + (3,))
    
    resampled_mask = ndi.map_coordinates(
        mask_data.astype(float),
        [mask_coords[..., 0], mask_coords[..., 1], mask_coords[..., 2]],
        order=0, mode='constant', cval=0, prefilter=False
    )
    return resampled_mask.astype(int)

def process_lesion_mask_watershed(suv_file, mask_file, output_file, threshold=2.5, z_boundary=2, 
                                 organ_file=None, organ_mapping=None):
    """
    Shrink lesion mask, exclude z-boundaries, subtract excluded organs (bladder), and separate using Watershed.
    """
    print(f"Loading SUV: {suv_file}")
    suv_img = nib.load(suv_file)
    suv_data = suv_img.get_fdata()
    
    print(f"Loading Mask: {mask_file}")
    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata().astype(int)
    
    target_affine = mask_img.affine
    target_header = mask_img.header
    
    if suv_data.shape != mask_data.shape:
        if not np.allclose(suv_img.affine, mask_img.affine, atol=1e-3):
            print(f"Warning: Affine mismatch detected! Resampling mask to SUV grid.")
            mask_data = resample_mask_to_reference(mask_data, mask_img.affine, suv_img.affine, suv_data.shape)
            target_affine = suv_img.affine
            target_header = suv_img.header
        else:
            print(f"Error: Shapes do not match and affines are similar! SUV: {suv_data.shape}, Mask: {mask_data.shape}")
            return
    elif not np.allclose(suv_img.affine, mask_img.affine, atol=1e-3):
        print(f"Warning: Affine mismatch detected (same shape)! Resampling mask to SUV grid.")
        mask_data = resample_mask_to_reference(mask_data, mask_img.affine, suv_img.affine, suv_data.shape)
        target_affine = suv_img.affine
        target_header = suv_img.header

    print(f"Starting processing: Threshold={threshold}, Z-Boundary={z_boundary}")
    
    # 1. Shrink
    print("  Shrinking mask based on SUV threshold...")
    shrunken_mask = (mask_data > 0) & (suv_data >= threshold)
    
    # 1.5 Z-boundary exclusion
    if z_boundary > 0:
        print(f"  Applying Z-boundary exclusion (removing {z_boundary} slices from ends)...")
        z_exclusion = create_z_boundary_exclusion_mask(shrunken_mask.shape, z_boundary)
        shrunken_mask[z_exclusion] = 0
    
    
    # 1.8 Exclusion Mask (Bladder)
    if organ_file and organ_mapping:
        if not os.path.exists(organ_file):
             print(f"  Warning: Organ mask file not found: {organ_file}")
        else:
             print(f"  Loading Organ Mask: {organ_file}")
        try:
            organ_img = nib.load(organ_file)
            organ_data = organ_img.get_fdata().astype(int)
            
            # Resample if needed to match SUV
            if organ_data.shape != suv_data.shape:
                 print("  Resampling organ mask to SUV grid...")
                 organ_data = resample_mask_to_reference(organ_data, organ_img.affine, suv_img.affine, suv_data.shape)
            
            print("  Applying exclusion mask (bladder)...")
            # Create exclusion mask (with SUV expansion if available)
            exclusion_mask = create_exclusion_mask(
                organ_data=organ_data, 
                organ_mapping=organ_mapping, 
                exclude_bladder=True, 
                suv_data=suv_data  # Enable smart SUV-based expansion
            )
            
            if np.any(exclusion_mask):
                cnt_before = np.sum(shrunken_mask)
                shrunken_mask[exclusion_mask] = 0
                cnt_after = np.sum(shrunken_mask)
                print(f"    Removed {cnt_before - cnt_after} voxels overlapping with excluded organs.")
            else:
                print("    No overlap with excluded organs found.")
                
        except Exception as e:
            print(f"  Error processing organ mask: {e}")

    # 2. Expand (Shrunk -> Expanded)
    # Allows recovering high-SUV areas adjacent to the shrunken core that might have been lost
    print(f"  Expanding mask (max dist {4})...")
    shrunken_mask = expand_mask_array(shrunken_mask, suv_data, threshold, max_distance=4)
    
    if np.sum(shrunken_mask) == 0:
        print("  Warning: No voxels remained after shrinking and boundary exclusion!")
        final_mask = np.zeros_like(mask_data, dtype=np.int32)
    else:
        # 3. Watershed Separation
        print("  Separating into individual lesions using Watershed...")
        
        # Initial CCL
        struct = connectivity_structure(18) # using 18 default
        initial_labels, n_init = ndi.label(shrunken_mask, structure=struct)
        print(f"  Initial connected blobs: {n_init}")
        
        split_labelmap = np.zeros_like(initial_labels, dtype=np.int32)
        next_global = 1
        
        for comp_id in range(1, int(initial_labels.max()) + 1):
            comp_mask = (initial_labels == comp_id)
            if not comp_mask.any():
                continue
                
            zsl, ysl, xsl = bbox_from_mask(comp_mask)
            comp_sub = comp_mask[zsl, ysl, xsl]
            suv_sub = suv_data[zsl, ysl, xsl]
            
            # Apply watershed splitting
            ws_sub = split_component_watershed(
                suv_sub=suv_sub,
                comp_mask_sub=comp_sub,
                component_id=comp_id,
                gaussian_sigma_voxels=2.0
            )
            
            # Paste results back
            for k in np.unique(ws_sub):
                if k == 0: continue
                sub = split_labelmap[zsl, ysl, xsl]
                sub[ws_sub == k] = next_global
                split_labelmap[zsl, ysl, xsl] = sub
                next_global += 1
                
        # Final CCL to guarantee contiguity
        final_mask = final_ccl_from_labelmap(split_labelmap, connectivity=18)
        
        unique_final = np.unique(final_mask)
        unique_final = unique_final[unique_final > 0]
        print(f"  Final Watershed separated lesions: {len(unique_final)}")
    
    print(f"Saving processed mask to: {output_file}")
    new_img = nib.Nifti1Image(final_mask.astype(np.int32), target_affine, target_header)
    nib.save(new_img, output_file)
    print("Done.")

def batch_process(results_dir, images_dir, threshold=2.5, z_boundary=2, organ_dir=None, mapping_file=None):
    """
    Batch process masks in results_dir using SUVs in images_dir.
    """
    organ_mapping = {}
    if mapping_file:
        organ_mapping = load_organ_mapping(mapping_file)
        if organ_mapping:
            print(f"Loaded {len(organ_mapping)} organ labels.")
    mask_pattern = os.path.join(results_dir, "*_t8_mask.nii.gz") 
    mask_files = glob.glob(mask_pattern)
    mask_files.sort()
    
    if not mask_files:
        print(f"No mask files found matching {mask_pattern}")
        return

    print(f"Found {len(mask_files)} masks to process in batch mode.")
    
    for mask_file in mask_files:
        filename = os.path.basename(mask_file)
        
        # logic to find SUV
        # pattern: xxx_t8_mask.nii.gz -> xxx_0001.nii.gz
        # or matches typical: xxx_mask.nii.gz
        
        if "_t8_mask.nii.gz" in filename:
            base_name = filename.replace("_t8_mask.nii.gz", "")
            suv_filename = f"{base_name}_0001.nii.gz"
            output_filename = filename.replace("_t8_mask.nii.gz", "_t8_mask_watershed.nii.gz")
        else:
            # Fallback/General
            base_name = filename.replace("_mask.nii.gz", "")
            suv_filename = f"{base_name}_0001.nii.gz" # Default assumption
            output_filename = filename.replace("_mask.nii.gz", "_mask_watershed.nii.gz")
            
        suv_file = os.path.join(images_dir, suv_filename)
        
        # logic to find organ file (if organ_dir provided)
        organ_file = None
        if organ_dir:
            # base_name usually: OSU107_acc0_400
            candidates = [
                os.path.join(organ_dir, f"{base_name}_400_0000_total.nii"),
            ]
            
            for cand in candidates:
                if os.path.exists(cand):
                    organ_file = cand
                    break
            
            if not organ_file:
                pass
        
        if not os.path.exists(suv_file):
            print(f"Warning: SUV file not found for {filename}: {suv_file}")
            # Try alternative if it didn't match specific pattern
            if "_t8_mask" not in filename: # maybe it has _400?
                 suv_filename_alt = f"{base_name}_400_0001.nii.gz"
                 suv_file_alt = os.path.join(images_dir, suv_filename_alt)
                 if os.path.exists(suv_file_alt):
                     suv_file = suv_file_alt
                     print(f"  Found SUV file with _400 suffix: {suv_file}")
                 else:
                     continue
            else:
                continue
                
        output_file = os.path.join(results_dir, output_filename)
        
        if os.path.exists(output_file):
            print(f"Skipping {output_filename} (already exists)")
            continue
            
        print(f"Processing {filename}...")
        try:
            process_lesion_mask_watershed(suv_file, mask_file, output_file, threshold, z_boundary, organ_file, organ_mapping)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Shrink & Separate lesion mask using Watershed.")
    
    parser.add_argument("--suv", "-s", help="Input PET SUV NIfTI file")
    parser.add_argument("--mask", "-m", help="Input Lesion Mask NIfTI file")
    parser.add_argument("--output", "-o", help="Output Processed Mask NIfTI file")
    
    parser.add_argument("--batch", "-b", action="store_true", help="Enable batch mode")
    parser.add_argument("--results_dir", "-r", 
                        default="nnunet/nnUNet_raw/Dataset001_autopet/labelsTr/nnunet_t8",
                        help="Directory containing mask files")
    parser.add_argument("--images_dir", "-i", 
                        default="nnunet/nnUNet_raw/Dataset001_autopet/imagesTr",
                        help="Directory containing SUV images")
    
    parser.add_argument("--organ_dir",  default="nnunet/nnUNet_raw/Dataset001_autopet/imagesTr",
                        help="Directory containing Organ segmentation files (totosegmentator file for exclusion)")
    parser.add_argument("--mapping_file", default=os.path.join(parent_dir, "data/totalsegmentator_index_mapping.json"),
                        help="JSON file mapping organ labels to names")
    
    parser.add_argument("--threshold", "-t", type=float, default=2.5, help="SUV threshold (default: 2.5)")
    parser.add_argument("--z_boundary", "-z", type=int, default=4, help="Z-boundary exclusion (default: 4)")
    
    args = parser.parse_args()
    
    if args.batch:
        if not os.path.exists(args.results_dir):
            print(f"Error: Results directory not found: {args.results_dir}")
            return
        if not os.path.exists(args.images_dir):
            print(f"Error: Images directory not found: {args.images_dir}")
            return
        batch_process(args.results_dir, args.images_dir, args.threshold, args.z_boundary, args.organ_dir, args.mapping_file)
    else:
        if not args.suv or not args.mask or not args.output:
            print("Error: Single mode requires --suv, --mask, and --output.")
            return
            
        mapping = {}
        # Parse single organ file
        organ_file = None
        if args.organ_dir:
            if not os.path.exists(args.organ_dir):
                print(f"Warning: Organ directory not found: {args.organ_dir}")
            
            if os.path.isfile(args.organ_dir):
                organ_file = args.organ_dir
            elif os.path.isdir(args.organ_dir):
                suv_basename = os.path.basename(args.suv)
                candidates = []
                
                if "0001.nii" in suv_basename:
                    candidates.append(suv_basename.replace("0001.nii.gz", "0000_total.nii"))
                
                for cand in candidates:
                    p = os.path.join(args.organ_dir, cand)
                    if os.path.exists(p):
                        organ_file = p
                        print(f"Auto-detected organ file: {organ_file}")
                        break
                        
                if not organ_file:
                    print(f"Warning: Could not find matching organ file in {args.organ_dir}")
                    print(f"Tried: {candidates}")
        
        # Load mapping
        if args.mapping_file:
             mapping = load_organ_mapping(args.mapping_file)
             
        process_lesion_mask_watershed(args.suv, args.mask, args.output, args.threshold, args.z_boundary, organ_file, mapping)

if __name__ == "__main__":
    main()
