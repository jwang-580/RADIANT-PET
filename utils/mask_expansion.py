# seperate lesions predicted from nnUNet and expand them based on SUV threshold (2.5) and max_distance(4 voxels)

import argparse
import numpy as np
import nibabel as nib
import os
import glob
from scipy import ndimage as ndi

def resample_mask_to_reference(mask_data, mask_affine, ref_affine, ref_shape):
    """
    Resample mask to match reference image voxel grid.
    """
    print("  Resampling mask to match reference affine...")
    
    # Create coordinate grids for the reference space
    i, j, k = np.mgrid[0:ref_shape[0], 0:ref_shape[1], 0:ref_shape[2]]
    
    # Convert reference voxel coordinates to homogeneous coordinates
    ref_voxel_coords = np.column_stack([
        i.ravel(), j.ravel(), k.ravel(), np.ones(i.size)
    ])
    
    # Transform reference voxels to RAS world coordinates
    ras_coords = ref_affine @ ref_voxel_coords.T
    
    # Transform RAS coordinates to mask voxel coordinates
    mask_affine_inv = np.linalg.inv(mask_affine)
    mask_voxel_coords = mask_affine_inv @ ras_coords
    
    # Extract just the spatial coordinates
    mask_coords = mask_voxel_coords[:3, :].T
    mask_coords = mask_coords.reshape(ref_shape + (3,))
    
    # Sample
    resampled_mask = ndi.map_coordinates(
        mask_data.astype(float),
        [mask_coords[..., 0], mask_coords[..., 1], mask_coords[..., 2]],
        order=0,  # Nearest neighbor
        mode='constant',
        cval=0,
        prefilter=False
    )
    
    return resampled_mask.astype(int)

def expand_mask_array(mask_data, suv_data, threshold=2.5, max_distance=4):
    """
    Expand mask array based on SUV threshold and connectivity.
    """
    # Define structure for 26-connectivity (3x3x3 box)
    struct = ndi.generate_binary_structure(3, 3)
    
    # Separate mask into individual lesions (connected components)
    # Ensure we treat any non-zero value as part of the mask
    binary_mask = mask_data > 0
    current_mask, num_features = ndi.label(binary_mask, structure=struct)
    print(f"  Found {num_features} individual lesions for expansion.")
    
    # If no lesions found, just return zeros
    if num_features == 0:
        return np.zeros_like(mask_data, dtype=int)
    
    for i in range(max_distance):
        # Dilate current mask to find neighbors
        dilated_labels = ndi.maximum_filter(current_mask, footprint=struct)
        
        # Identify candidates: background voxels, neighbor to mask, high SUV
        candidates_mask = (current_mask == 0) & (dilated_labels > 0) & (suv_data >= threshold)
        
        num_new_voxels = np.sum(candidates_mask)
        
        if num_new_voxels == 0:
            break
            
        current_mask[candidates_mask] = dilated_labels[candidates_mask]
        
    return (current_mask > 0).astype(int)

def expand_lesion_mask(suv_file, mask_file, output_file, threshold=2.5, max_distance=4):
    """
    Expand lesion mask to surroundings until SUV < threshold or max_distance reached.
    """
    print(f"Loading SUV: {suv_file}")
    suv_img = nib.load(suv_file)
    suv_data = suv_img.get_fdata()
    
    print(f"Loading Mask: {mask_file}")
    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata().astype(int)
    
    if suv_data.shape != mask_data.shape:
        # Check if it's just an affine mismatch that we can fix
        if not np.allclose(suv_img.affine, mask_img.affine, atol=1e-3):
            print(f"Warning: Affine mismatch detected! Resampling mask to SUV grid.")
            mask_data = resample_mask_to_reference(mask_data, mask_img.affine, suv_img.affine, suv_data.shape)
        else:
            print(f"Error: Shapes do not match and affines are similar! SUV: {suv_data.shape}, Mask: {mask_data.shape}")
            return
            
    # Double check affine even if shapes match (could be same shape but different orientation)
    elif not np.allclose(suv_img.affine, mask_img.affine, atol=1e-3):
        print(f"Warning: Affine mismatch detected (same shape)! Resampling mask to SUV grid.")
        mask_data = resample_mask_to_reference(mask_data, mask_img.affine, suv_img.affine, suv_data.shape)

    print(f"Starting expansion: Threshold={threshold}, Max Distance={max_distance} voxels")
    
    expanded_mask = expand_mask_array(mask_data, suv_data, threshold, max_distance)
        
    print(f"Saving expanded mask to: {output_file}")
    new_img = nib.Nifti1Image(expanded_mask, mask_img.affine, mask_img.header)
    nib.save(new_img, output_file)
    print("Done.")

def batch_expand(results_dir, images_dir, threshold=2.5, max_distance=4):
    """
    Batch expand masks in results_dir using SUVs in images_dir.
    """
    # Find all mask files
    mask_pattern = os.path.join(results_dir, "*_gt.nii.gz")
    mask_files = glob.glob(mask_pattern)
    mask_files.sort()
    
    if not mask_files:
        print(f"No mask files found matching {mask_pattern}")
        return

    print(f"Found {len(mask_files)} masks to process in batch mode.")
    
    for mask_file in mask_files:
        filename = os.path.basename(mask_file)
        # Extract ID (e.g., OSU02)
        # filename is xxx_gt.nii.gz
        # SUV filename: xxx_0001.nii.gz
        
        base_name = filename.replace("_gt.nii.gz", "") # xxx
        suv_filename = f"{base_name}_0001.nii.gz"
        suv_file = os.path.join(images_dir, suv_filename)
        
        if not os.path.exists(suv_file):
            print(f"Warning: SUV file not found for {filename}: {suv_file}")
            continue
            
        # Construct output filename
        output_filename = filename.replace("_gt.nii.gz", "_gt_dilate.nii.gz")
        output_file = os.path.join(results_dir, output_filename)
        
        if os.path.exists(output_file):
            print(f"Skipping {output_filename} (already exists)")
            continue
            
        print(f"Processing {filename}...")
        try:
            expand_lesion_mask(suv_file, mask_file, output_file, threshold, max_distance)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Expand lesion mask based on SUV threshold and distance.")
    
    # Single file mode arguments
    parser.add_argument("--suv", "-s", help="Input PET SUV NIfTI file (Single mode)")
    parser.add_argument("--mask", "-m", help="Input Lesion Mask NIfTI file (Single mode)")
    parser.add_argument("--output", "-o", help="Output Expanded Mask NIfTI file (Single mode)")
    
    # Batch mode arguments
    parser.add_argument("--batch", "-b", action="store_true", help="Enable batch mode")
    parser.add_argument("--results_dir", "-r", 
                        default="autopet-3-submission/nnUNet_raw/Dataset002_OSU/results",
                        help="Directory containing mask files (Batch mode)")
    parser.add_argument("--images_dir", "-i", 
                        default="autopet-3-submission/nnUNet_raw/Dataset002_OSU/imagesTr",
                        help="Directory containing SUV images (Batch mode)")
    
    # Common arguments
    parser.add_argument("--threshold", "-t", type=float, default=2.5, help="SUV threshold for expansion (default: 2.5)")
    parser.add_argument("--distance", "-d", type=int, default=3, help="Maximum expansion distance in voxels (default: 3)")
    
    args = parser.parse_args()
    
    if args.batch:
        if not os.path.exists(args.results_dir):
            print(f"Error: Results directory not found: {args.results_dir}")
            return
        if not os.path.exists(args.images_dir):
            print(f"Error: Images directory not found: {args.images_dir}")
            return
        batch_expand(args.results_dir, args.images_dir, args.threshold, args.distance)
    else:
        # Single file mode
        if not args.suv or not args.mask or not args.output:
            print("Error: In single mode, --suv, --mask, and --output are required.")
            print("Use --batch to run in batch mode.")
            return
            
        if not os.path.exists(args.suv):
            print(f"Error: SUV file not found: {args.suv}")
            return
        if not os.path.exists(args.mask):
            print(f"Error: Mask file not found: {args.mask}")
            return
            
        expand_lesion_mask(args.suv, args.mask, args.output, args.threshold, args.distance)

if __name__ == "__main__":
    main()
