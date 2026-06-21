import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, distance
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, relabel_sequential
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from scipy.ndimage import map_coordinates, generate_binary_structure
from skimage import morphology
from utils.mask_expansion import expand_mask_array


@dataclass
class LesionShape:
    surface_area_voxels: float
    sphericity: float
    surface_to_volume: float
    max_diameter_voxels: float
    elongation: float
    flatness: float
    solidity: float


def convert_numpy_types(obj):
    """
    Recursively convert NumPy data types to Python native types for JSON serialization.
    
    Args:
        obj: Any object that may contain NumPy types
        
    Returns:
        Object with NumPy types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def connectivity_structure(connectivity: int = 18) -> np.ndarray:
    """Create connectivity structure for connected component analysis"""
    if connectivity == 6:
        return ndi.generate_binary_structure(3, 1)
    if connectivity == 26:
        return np.ones((3,3,3), dtype=bool)
    s = np.ones((3,3,3), dtype=bool)  # 18-connectivity: remove corners
    for z in (0,2):
        for y in (0,2):
            for x in (0,2):
                s[z,y,x] = False
    return s


def bbox_from_mask(mask3d: np.ndarray) -> Tuple[slice, slice, slice]:
    """Extract bounding box from 3D mask"""
    z, y, x = np.where(mask3d)
    return slice(z.min(), z.max()+1), slice(y.min(), y.max()+1), slice(x.min(), x.max()+1)


def final_ccl_from_labelmap(labelmap: np.ndarray, connectivity: int = 18) -> np.ndarray:
    """Final connected component labeling with guaranteed contiguity"""
    struct = connectivity_structure(connectivity)
    out = np.zeros_like(labelmap, dtype=np.int32)
    cur = 1
    for lab in np.unique(labelmap):
        if lab == 0: 
            continue
        cc, n = ndi.label(labelmap == lab, structure=struct)
        for k in range(1, n+1):
            out[cc == k] = cur
            cur += 1
    return out


def filter_small_components(labelmap: np.ndarray, min_voxels: int = 20) -> Tuple[np.ndarray, int]:
    """Remove labels with fewer than min_voxels; returns (filtered_labelmap, n_removed)"""
    maxlab = int(labelmap.max())
    if maxlab == 0:
        return labelmap, 0
    
    counts = np.bincount(labelmap.ravel(), minlength=maxlab+1)
    keep = np.ones(maxlab+1, dtype=bool)
    keep[0] = True  # background stays
    keep &= (counts >= min_voxels)
    
    drop = np.where(~keep)[0]
    drop = drop[drop != 0]  # exclude background
    
    out = labelmap.copy()
    if drop.size:
        out[np.isin(out, drop)] = 0
    
    n_removed = int(drop.size)
    return out, n_removed


def compute_shape_metrics(mask3d: np.ndarray) -> LesionShape:
    """Compute shape metrics for a 3D mask"""
    voxels = int(mask3d.sum())

    # Surface area in voxels (8-connected surface)
    struct = ndi.generate_binary_structure(3, 1)
    eroded = ndi.binary_erosion(mask3d, structure=struct, iterations=1, border_value=0)
    surface = mask3d & ~eroded
    surface_voxels = int(surface.sum())

    # Surface to volume ratio
    s2v = surface_voxels / voxels if voxels > 0 else 0.0

    # Sphericity in voxel space
    if voxels > 0 and surface_voxels > 0:
        # Sphere with same volume would have surface = (36π)^(1/3) * V^(2/3)
        ideal_surface = (36 * np.pi) ** (1/3) * (voxels ** (2/3))
        sphericity = ideal_surface / surface_voxels if surface_voxels > 0 else 0.0
    else:
        sphericity = 0.0

    # Get voxel coordinates (no mm conversion)
    coords = np.column_stack(np.where(mask3d))
    if coords.shape[0] > 20000:
        coords = coords[np.random.choice(coords.shape[0], 20000, replace=False)]

    # Convex hull and max diameter in voxel space
    if coords.shape[0] >= 4:
        try:
            hull = ConvexHull(coords)
            hull_pts = coords[hull.vertices]
        except Exception:
            hull_pts = coords
    else:
        hull_pts = coords

    if hull_pts.shape[0] >= 2:
        max_diam = float(distance.pdist(hull_pts, metric='euclidean').max())
    else:
        max_diam = 0.0

    # Eigenanalysis for elongation & flatness
    if coords.shape[0] >= 3:
        C = np.cov(coords.T)
        evals = np.sort(np.linalg.eigvalsh(C))[::-1]
        if evals[0] > 0:
            elong = float(np.sqrt(max(evals[1],0)/evals[0]))
            flat  = float(np.sqrt(max(evals[2],0)/evals[0]))
        else:
            elong = flat = 0.0
    else:
        elong = flat = 0.0

    # Solidity (convex hull volume in voxel space)
    try:
        if hull_pts.shape[0] >= 4:
            Vhull = float(ConvexHull(hull_pts).volume)
            solidity = float(voxels / Vhull) if Vhull > 0 else 0.0
        else:
            solidity = 0.0
    except Exception:
        solidity = 0.0

    return LesionShape(
        surface_area_voxels=surface_voxels,
        sphericity=sphericity,
        surface_to_volume=s2v,
        max_diameter_voxels=max_diam,
        elongation=elong,
        flatness=flat,
        solidity=solidity
    )


def split_component_watershed(
    suv_sub: np.ndarray,
    comp_mask_sub: np.ndarray,
    spacing_zyx=(1.0, 1.0, 1.0),
    gaussian_sigma_voxels=2.0,  # Increased default smoothing from 1.0 to 2.0
    component_id: int = 0,
) -> np.ndarray:
    """
    1) Distance-based watershed to split a blob.
    2) Validate that each separating boundary passes through an SUV valley:
        For each pair (i, j):  (max_peak(i,j) - saddleSUV(i,j)) / max_peak(i,j)  >= tau
    If ANY pair fails, reject the split: return a single component.
    3) For any component size larger than 1000 voxels, perform additional SUV-based watershed splitting.

    Returns: label map (same shape as comp_mask_sub). If rejected, it's 1 inside mask, 0 outside.
    """
    
    if not np.any(comp_mask_sub):
        return np.zeros_like(comp_mask_sub, np.int32)

    comp_volume = np.sum(comp_mask_sub)
    
    # Simple parameters
    sigma = gaussian_sigma_voxels
    
    # Early exit for very small components
    if comp_volume < 20:  # Simple threshold
        out = np.zeros_like(comp_mask_sub, np.int32)
        out[comp_mask_sub] = 1
        return out

    # Simple distance transform
    dist = ndi.distance_transform_edt(comp_mask_sub.astype(bool), sampling=spacing_zyx)
        
    dist_s = ndi.gaussian_filter(dist, sigma=sigma) if sigma > 0 else dist

    # Very aggressive distance-based peak detection - find ALL possible splits
    coords = peak_local_max(
        dist_s,
        labels=comp_mask_sub.astype(np.uint8),
        min_distance=8,  # Increased to 10 for very conservative splitting
        footprint=np.ones((3,3,3), bool),
        exclude_border=False
    )

    # If <2 markers → nothing to split
    if coords.shape[0] < 2:
        out = np.zeros_like(comp_mask_sub, np.int32) 
        out[comp_mask_sub] = 1 
        return out

    # Allow markers but be more conservative
    max_markers = min(30, comp_volume // 100)  # Reduced cap significantly
    if len(coords) > max_markers:
        # Keep the strongest distance peaks
        peak_strengths = [dist_s[tuple(coord)] for coord in coords]
        top_indices = np.argsort(peak_strengths)[-max_markers:]
        coords = coords[top_indices]

    markers = np.zeros_like(comp_mask_sub, np.int32)
    for i, (z,y,x) in enumerate(coords, start=1):
        markers[z,y,x] = i

    # Apply watershed without mask to avoid losing unreachable voxels
    labels = watershed(-dist_s, markers=markers, mask=comp_mask_sub.astype(bool))
    
    # Check watershed output before relabeling
    unique_before = np.unique(labels)
    unique_before = unique_before[unique_before > 0]  # Remove background
    
    
    labels, _, _ = relabel_sequential(labels)
    
    # Check after relabeling
    unique_after = np.unique(labels)
    unique_after = unique_after[unique_after > 0]  # Remove background

    # If watershed still produced <2 regions, reject (single)
    if labels.max() < 2:
        out = np.zeros_like(comp_mask_sub, np.int32) 
        out[comp_mask_sub] = 1 
        return out

    # --- Validate SUV valley at each boundary; reject if any boundary is shallow ---
    suv = suv_sub.astype(np.float32)

    # Precompute per-label SUV peaks and means for better validation
    peaks = {}
    means = {}
    for lab in range(1, labels.max()+1):
        m = (labels == lab)
        if np.any(m):
            peaks[lab] = float(np.max(suv[m]))
            means[lab] = float(np.mean(suv[m]))
        else:
            peaks[lab] = 0.0
            means[lab] = 0.0

    # Find adjacency (6-neighborhood) and estimate "saddle" SUV along shared borders
    shifts = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    # Improved border mask using proper 6-connectivity erosion
    struct_6conn = ndi.generate_binary_structure(3, 1)  # 6-connectivity
    shell = (labels > 0) & ~ndi.binary_erosion(labels > 0, structure=struct_6conn)

    def touching_pairs(a, b):
        m = (a!=0) & (b!=0) & (a!=b)
        if not np.any(m): return {}
        p = np.sort(np.stack([a[m], b[m]], 1), 1)
        u, c = np.unique(p, axis=0, return_counts=True)
        return {tuple(k): int(v) for k,v in zip(u, c)}

    pairs_total = {}
    for dz,dy,dx in shifts:
        neigh = np.roll(np.roll(np.roll(labels, dz,0), dy,1), dx,2)
        for k,v in touching_pairs(labels, neigh).items():
            pairs_total[k] = pairs_total.get(k, 0) + v

    # Check each boundary's valley depth 
    for (i, j), contact_count in pairs_total.items():
        # Require minimum contact for reliable boundary analysis
        if contact_count < 3:
            continue
            
        # Gather border voxels of i touching j (and vice versa)
        saddle_vals = []
        for dz,dy,dx in shifts:
            neigh = np.roll(np.roll(np.roll(labels, dz,0), dy,1), dx,2)
            m_ij = (labels==i) & shell & (neigh==j)
            if np.any(m_ij): 
                # Use mean of boundary values instead of max for more stable saddle estimate
                saddle_vals.extend(suv[m_ij].tolist())
            m_ji = (labels==j) & shell & (neigh==i)
            if np.any(m_ji): 
                saddle_vals.extend(suv[m_ji].tolist())

        if len(saddle_vals) < 2:
            # insufficient boundary samples; be conservative and reject
            out = np.zeros_like(comp_mask_sub, np.int32)
            out[comp_mask_sub] = 1
            return out

        # Use median of saddle values for robustness against outliers
        saddle = float(np.median(saddle_vals))
        peakij = max(peaks[i], peaks[j])
        
        if peakij <= saddle:
            # degenerate case - no meaningful peak; reject
            out = np.zeros_like(comp_mask_sub, np.int32)
            out[comp_mask_sub] = 1
            return out

        rel_drop = (peakij - saddle) / peakij
        
        # Adaptive valley validation based on SUV uptake level
        # Higher SUV lesions require stricter validation (more reliable signal)
        # Lower SUV lesions use more lenient thresholds (capture subtle separations)
        if peakij >= 10.0:
            required_drop = 0.75  # Very strict for high SUV lesions
        else:
            required_drop = 0.6   # Strict for lower/moderate SUV lesions
        
        if rel_drop < required_drop:
            # Boundary not a real SUV valley or regions too similar → reject the distance-based split
            # But still try SUV-based splitting for large components
            if comp_volume > 1000:
                suv_split = suv_based_watershed_large_component(
                    suv_sub, comp_mask_sub, comp_volume, None, sigma
                )
                if suv_split is not None and suv_split.max() >= 2:
                    return suv_split
            
            out = np.zeros_like(comp_mask_sub, np.int32)
            out[comp_mask_sub] = 1
            return out

    # All boundaries pass the valley test → proceed to step 3
    
    # --- Step 3: For large regions within the validated split, attempt additional SUV-based splitting ---
    # Check each region from step 2 - if any region is still >1000 voxels, try to split it further
    final_labels = labels.copy()
    next_label = labels.max() + 1
    
    for region_id in range(1, labels.max() + 1):
        region_mask = (labels == region_id)
        region_volume = np.sum(region_mask)
        
        if region_volume > 1000:  # This individual region is still too large
            # Extract this region for SUV-based splitting
            region_coords = np.where(region_mask)
            region_bbox = (
                slice(region_coords[0].min(), region_coords[0].max() + 1),
                slice(region_coords[1].min(), region_coords[1].max() + 1), 
                slice(region_coords[2].min(), region_coords[2].max() + 1)
            )
            
            region_suv_sub = suv_sub[region_bbox]
            region_mask_sub = region_mask[region_bbox] 
            
            # Try SUV-based splitting on this individual region
            suv_split = suv_based_watershed_large_component(
                region_suv_sub, region_mask_sub, region_volume, None, sigma
            )
            
            # If SUV splitting produced multiple regions, replace the original region
            if suv_split is not None and suv_split.max() >= 2:
                candidate = final_labels.copy()
                candidate[region_mask] = 0

                filled_any = False
                for sub_region_id in range(1, suv_split.max() + 1):
                    sub_region_mask = (suv_split == sub_region_id)
                    if not np.any(sub_region_mask):
                        continue
                    full_mask = np.zeros_like(candidate, dtype=bool)
                    full_mask[region_bbox] = sub_region_mask
                    candidate[full_mask] = next_label
                    next_label += 1
                    filled_any = True

                # safety fallback: if anything remains 0 inside the original region, revert
                if filled_any and np.all(candidate[region_mask] > 0):
                    final_labels = candidate
                # else: keep the original 'final_labels' (no zeroing)

                # Any accidental holes inside the mask? Restore original labels there
                holes = comp_mask_sub & (final_labels == 0)
                if np.any(holes):
                    final_labels[holes] = labels[holes]
    
    # Final debug output
    final_unique = np.unique(final_labels)
    final_unique = final_unique[final_unique > 0]
    
    return final_labels


def suv_based_watershed_large_component(suv_sub: np.ndarray, comp_mask_sub: np.ndarray, 
                                       region_volume: int, existing_labels: np.ndarray, sigma: float) -> np.ndarray:
    """
    Aggressive SUV valley-based splitting for large components.
    - Seeds densely from SUV peaks.
    - Validates splits using the watershed-line (saddle) SUV.
    - Adaptive, lenient acceptance: passes on relative OR absolute drop (noise-aware).
    - Iteratively merges only clearly weak interfaces; keeps subtle SUV-only separations.
    """
    # Smooth SUV with provided sigma
    suv = suv_sub.astype(np.float32)
    suv_smooth = ndi.gaussian_filter(suv, sigma=float(sigma))

    if not np.any(comp_mask_sub):
        return np.zeros_like(comp_mask_sub, np.int32)

    # ----- Dense, adaptive peak seeding (aggressive) -----
    vals = suv_smooth[comp_mask_sub]
    if vals.size == 0:
        return (existing_labels if existing_labels is not None 
                else np.where(comp_mask_sub, 1, 0).astype(np.int32))
    # Lower the bar: keep modest peaks too
    # Valid peaks only
    thr_abs = max(3.0, float(np.percentile(vals, 0.80)))  # 80th percentile
    
    suv_peaks = peak_local_max(
        suv_smooth,
        labels=comp_mask_sub.astype(np.uint8),
        min_distance=10,                        
        threshold_abs=thr_abs,
        footprint=np.ones((3,3,3), bool),
        exclude_border=False
    )

    if suv_peaks.shape[0] < 2:
        if existing_labels is not None:
            return existing_labels
        out = np.zeros_like(comp_mask_sub, np.int32); out[comp_mask_sub] = 1
        return out

    # Sort by intensity and allow more markers
    peak_intensities = [suv_smooth[tuple(p)] for p in suv_peaks]
    suv_peaks = suv_peaks[np.argsort(peak_intensities)[::-1]]
    max_markers = max(2, min(100, region_volume // 30))  # higher cap, denser seeding
    if suv_peaks.shape[0] > max_markers:
        suv_peaks = suv_peaks[:max_markers]

    # Place markers
    suv_markers = np.zeros_like(comp_mask_sub, np.int32)
    for k, (z, y, x) in enumerate(suv_peaks, start=1):
        suv_markers[int(z), int(y), int(x)] = k

    struct6 = ndi.generate_binary_structure(3, 1)

    # ----- Iteratively prune only clearly weak interfaces -----
    # (Aggressive: lenient thresholds; keep subtle valleys)
    for _ in range(64):  # safety bound
        labels = watershed(-suv, markers=suv_markers, mask=comp_mask_sub)
        if labels.max() < 2:
            # nothing to split
            if existing_labels is not None:
                return existing_labels
            out = np.zeros_like(comp_mask_sub, np.int32); out[comp_mask_sub] = 1
            return out

        # Watershed line to sample the actual saddle
        labels_line = watershed(-suv, markers=suv_markers, mask=comp_mask_sub, watershed_line=True)
        ws_line = (labels_line == 0) & comp_mask_sub

        present = [lab for lab in range(1, labels.max()+1) if np.any(labels == lab)]
        peaks = {lab: float(np.max(suv_smooth[labels == lab])) for lab in present}
        vols  = {lab: int((labels == lab).sum()) for lab in present}

        merged_any = False
        for i in present:
            if merged_any:
                break
            di = ndi.binary_dilation(labels == i, structure=struct6)
            for j in present:
                if j <= i:
                    continue
                dj = ndi.binary_dilation(labels == j, structure=struct6)
                bij = ws_line & di & dj
                if np.count_nonzero(bij) < 5:
                    continue  # not enough boundary evidence

                # Robust saddle estimate on the boundary (median)
                saddle = float(np.median(suv_smooth[bij]))

                peakij = max(peaks.get(i, 0.0), peaks.get(j, 0.0))
                if peakij <= 0.0:
                    # degenerate; drop the smaller region's seed
                    src, tgt = (i, j) if vols.get(i, 0) < vols.get(j, 0) else (j, i)
                    pos = np.argwhere(suv_markers == src)
                    if pos.size: suv_markers[tuple(pos[0])] = 0
                    merged_any = True
                    break

                abs_drop = peakij - saddle
                rel_drop = abs_drop / max(peakij, 1e-6)

                if peakij > 10.0:
                    required_drop = 0.6  # Very strict for high SUV
                else:
                    required_drop = 0.6   # Strict for moderate SUV

                passes = (rel_drop >= required_drop)

                if not passes:
                    # clearly weak interface → merge smaller into larger (remove its seed)
                    src, tgt = (i, j) if vols.get(i, 0) < vols.get(j, 0) else (j, i)
                    pos = np.argwhere(suv_markers == src)
                    if pos.size: suv_markers[tuple(pos[0])] = 0
                    merged_any = True
                    break

        if not merged_any:
            break  # all remaining interfaces acceptable

    # Final relabel
    final_labels = watershed(-suv, markers=suv_markers, mask=comp_mask_sub)
    final_labels, _, _ = relabel_sequential(final_labels)

    if final_labels.max() < 1:
        out = np.zeros_like(comp_mask_sub, np.int32); out[comp_mask_sub] = 1
        return out

    return final_labels


def resample_mask_to_suv_grid(mask_data: np.ndarray, mask_affine: np.ndarray, 
                             suv_affine: np.ndarray, suv_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Resample a single mask to match SUV voxel grid. (because mask and organ have different voxel sizes)
    
    Args:
        mask_data: The mask data array to resample
        mask_affine: The affine matrix of the mask
        suv_affine: The affine matrix of the SUV image
        suv_shape: Shape of the SUV image
        
    Returns:
        Resampled mask data array
    """
    # Create coordinate grids for the SUV space
    i, j, k = np.mgrid[0:suv_shape[0], 0:suv_shape[1], 0:suv_shape[2]]
    
    # Convert SUV voxel coordinates to homogeneous coordinates
    suv_voxel_coords = np.column_stack([
        i.ravel(), j.ravel(), k.ravel(), np.ones(i.size)
    ])
    
    # Transform SUV voxels to RAS world coordinates
    ras_coords = suv_affine @ suv_voxel_coords.T
    
    # Transform RAS coordinates to mask voxel coordinates
    mask_affine_inv = np.linalg.inv(mask_affine)
    mask_voxel_coords = mask_affine_inv @ ras_coords
    
    # Extract just the spatial coordinates (drop homogeneous coordinate)
    mask_coords = mask_voxel_coords[:3, :].T
    mask_coords = mask_coords.reshape(suv_shape + (3,))
    
    # Use map_coordinates to sample mask data at the computed coordinates
    resampled_mask = map_coordinates(
        mask_data.astype(float),
        [mask_coords[..., 0], mask_coords[..., 1], mask_coords[..., 2]],
        order=0,  # Nearest neighbor for label data
        mode='constant',
        cval=0,  # Background value
        prefilter=False
    )
    
    return resampled_mask.astype(int)


def compute_liver_suv(organ_data: np.ndarray, suv_data: np.ndarray, liver_label: int = 5) -> Dict:
    """
    Compute robust liver SUVmax using liver organ mask.
    Uses median ± 2.5 × 1.4826 × MAD filtering and 5th-95th percentile trimming.
    
    Args:
        organ_data: Organ segmentation data
        suv_data: SUV image data
        liver_label: Label for liver in organ data (default 5)
        
    Returns:
        Dictionary with liver SUV statistics and threshold value
    """
    # Find liver voxels
    liver_mask = (organ_data == liver_label)
    liver_suv_values = suv_data[liver_mask]
    
    if liver_suv_values.size == 0:
        print(f"Warning: No liver voxels found (organ label {liver_label}). Using default SUV = 3.0")
        return {
            'liver_suv_threshold': 3.0,
            'median': 0.0,
            'mad': 0.0,
            'lower_bound': 0.0,
            'upper_bound': 3.0,
            'total_liver_voxels': 0,
            'robust_voxels_count': 0
        }
    
    # Step 1: Apply percentile trimming (5th-95th percentile)
    p5, p95 = np.percentile(liver_suv_values, [5, 95])
    trimmed_values = liver_suv_values[(liver_suv_values >= p5) & (liver_suv_values <= p95)]

    # Step 2: Apply median ± 2.5 × 1.4826 × MAD filtering
    median = np.median(trimmed_values)
    mad = np.median(np.abs(trimmed_values - median))  # Median Absolute Deviation
    
    # Keep voxels within median ± 2.5 × 1.4826 × MAD
    threshold = 2.5 * 1.4826 * mad
    lower_bound = median - threshold
    upper_bound = median + threshold
    
    robust_values = trimmed_values[
        (trimmed_values >= lower_bound) & (trimmed_values <= upper_bound)
    ]
    
    if robust_values.size == 0:
        print("Warning: No liver voxels remain after robust filtering. Using default SUV = 3.0")
        liver_suv = 3.0
    else:
        liver_suv = float(upper_bound)
    
    return {
        'liver_suv_threshold': float(liver_suv),
        'median': float(median),
        'mad': float(mad),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'total_liver_voxels': int(liver_suv_values.size),
        'robust_voxels_count': int(robust_values.size) if robust_values.size > 0 else 0
    }


def lesion_surface(lesion_mask: np.ndarray) -> np.ndarray:
    """Return 6-connected surface voxels of a lesion mask."""
    struct = ndi.generate_binary_structure(3, 1)  # 6-connectivity
    eroded = ndi.binary_erosion(lesion_mask, structure=struct, iterations=1, border_value=0)
    return lesion_mask & ~eroded


def spacing_zyx_from_voxel_size(voxel_size: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Convert voxel size to spacing in (z, y, x) order for scipy functions.
    
    Args:
        voxel_size: [dim0, dim1, dim2] from NIfTI header
                   typically dim0=x(R-L), dim1=y(A-P), dim2=z(I-S)
    
    Returns:
        Spacing in (z, y, x) order
    """
    return (voxel_size[2], voxel_size[1], voxel_size[0])


def calculate_spatial_relationship(lesion_center: np.ndarray, organ_mask: np.ndarray, 
                                 image_shape: Tuple[int, int, int]) -> Dict[str, any]:
    """
    Calculate spatial relationship between lesion and closest organ surface.
    
    Args:
        lesion_center: [x, y, z] coordinates of lesion center
        organ_mask: Boolean mask of the organ
        image_shape: Shape of the image for centering calculations
        
    Returns:
        Dictionary with spatial relationship descriptors
    """
    if not np.any(organ_mask):
        return {"relative_position": "unknown"}
    
    # Calculate organ center coordinates
    organ_coords = np.where(organ_mask)
    organ_center_raw = np.array([
        np.mean(organ_coords[0]),  # x center (dim0)
        np.mean(organ_coords[1]),  # y center (dim1)
        np.mean(organ_coords[2])   # z center (dim2)
    ])
    
    # Apply same centering as center_coords for consistency
    image_center_x = image_shape[0] / 2.0  # Center of x-axis (dim0)
    image_center_y = image_shape[1] / 2.0  # Center of y-axis (dim1)
    image_center_z = image_shape[2] / 2.0  # Center of z-axis (dim2)
    
    organ_center_centered = np.array([
        organ_center_raw[0] - image_center_x,  # x centered
        organ_center_raw[1] - image_center_y,  # y centered
        organ_center_raw[2] - image_center_z   # z centered
    ])
    
    # Find closest point on organ surface to lesion center
    organ_points = np.column_stack(organ_coords)  # Shape: (N, 3) - [x, y, z] coordinates
    lesion_point = lesion_center.reshape(1, 3)    # Shape: (1, 3) - [x, y, z]
    
    # Calculate Euclidean distances
    distances = np.sqrt(np.sum((organ_points - lesion_point)**2, axis=1))
    closest_idx = np.argmin(distances)
    
    # Get coordinates of closest organ surface point  
    closest_organ_point_raw = organ_points[closest_idx]  # This is [x, y, z] from np.column_stack
    
    # Apply centering for consistency (keep x, y, z format)
    closest_organ_point_centered = np.array([
        closest_organ_point_raw[0] - image_center_x,  # x centered
        closest_organ_point_raw[1] - image_center_y,  # y centered
        closest_organ_point_raw[2] - image_center_z   # z centered
    ])
    
    # Calculate direction vector from closest organ surface to lesion center
    direction_vector = lesion_center - closest_organ_point_raw  # Keep using raw for distance calculation
    
    # Medical imaging coordinate system:
    # np.where returns (dim0_indices, dim1_indices, dim2_indices) = (x, y, z)
    # For medical images in RAS+ orientation:
    # x (dim0): Left(+) to Right(-) - first spatial dimension; left and right is opposite on CT
    # y (dim1): Posterior(-) to Anterior(+) - second spatial dimension  
    # z (dim2): Inferior(-) to Superior(+) - third spatial dimension (axial/head-foot)
    
    directions = []
    
    # Left-Right relationship (x-axis)
    if direction_vector[0] > 0:
        directions.append("left")
    elif direction_vector[0] == 0:
        directions.append("same x-axis")
    else:
        directions.append("right")
    
    # Posterior-Anterior relationship (y-axis)
    if direction_vector[1] > 0:
        directions.append("anterior")
    elif direction_vector[1] == 0:
        directions.append("same y-axis")
    else:
        directions.append("posterior")
            
    # Superior-Inferior relationship (z-axis)
    if direction_vector[2] > 0:
        directions.append("superior")
    elif direction_vector[2] == 0:
        directions.append("same z-axis")
    else:
        directions.append("inferior")
    
    # Create descriptive relative position
    if len(directions) == 0:
        relative_position = "adjacent"
    elif len(directions) == 1:
        relative_position = directions[0]
    else:
        # Combine directions (e.g., "superior-right", "anterior-left")
        relative_position = "-".join(directions)
    
    return {
        "relative_position": relative_position,
        # "distance_vector": {
        #     "x_voxels": float(direction_vector[0]),      # Left(+) to Right(-)
        #     "y_voxels": float(direction_vector[1]),      # Posterior(-) to Anterior(+)
        #     "z_voxels": float(direction_vector[2])       # Inferior(-) to Superior(+)
        # },
        "surface_to_lesion_distance_voxels": float(distances[closest_idx]),
        # "organ_center_coordinates": {
        #     "x": float(organ_center_centered[0]),  # x coordinate (centered)
        #     "y": float(organ_center_centered[1]),  # y coordinate (centered)
        #     "z": float(organ_center_centered[2])   # z coordinate (centered)
        # },
        # "closest_organ_surface_coordinates": {
        #     "x": float(closest_organ_point_centered[0]),  # x coordinate (centered)
        #     "y": float(closest_organ_point_centered[1]),  # y coordinate (centered)
        #     "z": float(closest_organ_point_centered[2])   # z coordinate (centered)
        # }
    }


def labels_by_contains(organ_mapping: Dict[int, str], substrs: List[str]) -> List[int]:
    """Find organ labels that contain any of the given substrings"""
    subs = [s.lower() for s in substrs]
    return [lab for lab, name in organ_mapping.items()
            if any(s in name.lower() for s in subs)]


def create_exclusion_mask(organ_data: np.ndarray, organ_mapping: Dict[int, str],
                         exclude_brain: bool = False, exclude_kidneys: bool = False, 
                         exclude_bladder: bool = False, dilate_voxels: int = 4,
                         suv_data: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Create exclusion mask for organs with physiological uptake.
    If suv_data is provided, performs smart expansion:
    1. Expand to include all connected voxels with SUV >= 4.0
    2. Then expand to include neighbors with SUV >= 2.5 (max 2 voxels distance)
    """
    exclusion_mask = np.zeros_like(organ_data, dtype=bool)
    
    label_sets = []
    if exclude_brain:
        label_sets.append(labels_by_contains(organ_mapping, ["brain"]))
    if exclude_kidneys:
        label_sets.append(labels_by_contains(organ_mapping, ["kidney"]))
        label_sets.append(labels_by_contains(organ_mapping, ["ureter", "renal pelvis"])) 
    if exclude_bladder:
        label_sets.append(labels_by_contains(organ_mapping, ["urinary_bladder"]))

    exclude_ids = sorted({lab for group in label_sets for lab in group})
    
    for lab in exclude_ids:
        exclusion_mask |= (organ_data == lab)

    initial_voxel_count = exclusion_mask.sum()

    if suv_data is not None:
        # Step 1: Expand to include all connected SUV >= 4.0
        # Use large max_distance to capture entire high-uptake regions
        print("    Expansion Step 1: SUV >= 4.0 (full connected)")
        exclusion_mask = expand_mask_array(exclusion_mask.astype(int), suv_data, threshold=4.0, max_distance=100)
        
        # Step 2: Expand to neighbors with SUV >= 2.5, max 2 voxels
        print("    Expansion Step 2: SUV >= 2.5 (max 2 voxels)")
        exclusion_mask = expand_mask_array(exclusion_mask.astype(int), suv_data, threshold=2.5, max_distance=2)
        
        exclusion_mask = exclusion_mask.astype(bool)

    # Optional morphological dilation (legacy or supplementary)
    # If SUV data was used, we might rely on it, but keeping this for robustness if requested
    if dilate_voxels > 0 and suv_data is None:
         # Only do blind dilation if no SUV data (or maybe we still want it? 
         # User instructions implied SUV expansion 'instead' or 'after' defining organ.
         # For safety, if SUV expansion happened, maybe skipping blind dilation is better to avoid over-masking?
         # Or maybe the user implies replacing the blind dilation with smart expansion.
         # "update ... to make it more robust: after defining organ ... expand on suv ... then ..."
         # I will assume smart expansion replaces blind dilation if suv_data is present.
         pass
    elif dilate_voxels > 0:
        exclusion_mask = ndi.binary_dilation(exclusion_mask, iterations=dilate_voxels)

    final_voxel_count = exclusion_mask.sum()

    print(f"Excluding {len(exclude_ids)} labels for physiological uptake: {exclude_ids}")
    print(f"  Final excluded voxels: {final_voxel_count:,}")
    return exclusion_mask


def create_z_boundary_exclusion_mask(image_shape: Tuple[int, int, int], 
                                    boundary_voxels: int = 2) -> np.ndarray:
    """
    Create a mask to exclude boundary regions along the Z-axis (head-to-foot direction)
    to remove edge artifacts and noise commonly found at scan boundaries
    
    Args:
        image_shape: Shape of the image (z, y, x)
        boundary_voxels: Number of voxels to exclude from each end of Z-axis
        
    Returns:
        Boolean mask where True indicates regions to exclude from segmentation
    """
    z_boundary_mask = np.zeros(image_shape, dtype=bool)
    
    # Get the Z-axis dimension (Dim2 based on our previous analysis)
    z_size = image_shape[2]  # Dim2 is the axial (head-to-foot) direction
    
    if boundary_voxels > 0 and z_size > 2 * boundary_voxels:
        # Exclude first 'boundary_voxels' slices (head end)
        z_boundary_mask[:, :, :boundary_voxels] = True
        
        # Exclude last 'boundary_voxels' slices (feet end)  
        z_boundary_mask[:, :, -boundary_voxels:] = True
        
        total_excluded_z_voxels = np.sum(z_boundary_mask)
    
    return z_boundary_mask


def analyze_organ_overlaps(organ_data: np.ndarray, organ_mapping: Dict[int, str], 
                          lesion_voxels: np.ndarray) -> Dict[str, float]:
    """
    Analyze overlap of lesion with surrounding organs 
    
    Args:
        organ_data: Organ segmentation data
        organ_mapping: Mapping from organ labels to names
        lesion_voxels: Boolean array indicating lesion locations
        
    Returns:
        Dictionary mapping organ names to overlap percentages (all overlaps)
    """
    # Get unique organ labels at lesion locations (vectorized)
    organ_labels_at_lesion = organ_data[lesion_voxels]
    unique_labels, counts = np.unique(organ_labels_at_lesion, return_counts=True)
    
    lesion_volume = lesion_voxels.sum()
    organ_overlaps = {}
    
    # Process overlaps in vectorized manner
    for label, count in zip(unique_labels, counts):
        if label == 0:  # Skip background
            continue
        
        if label in organ_mapping:
            organ_name = organ_mapping[label]
            overlap_percentage = (count / lesion_volume) * 100
            organ_overlaps[organ_name] = overlap_percentage
    
    # Return all overlaps (will be filtered later in combine_organ_results)
    return organ_overlaps


def combine_organ_results(organ_overlaps: Dict[str, float], closest_organs: Dict[str, Dict[str, float]], 
                         target_count: int = 3) -> Dict[str, Dict]:
    """
    Combine organ overlaps and closest organs to get exactly target_count total organs
    Prioritizes overlaps over closest organs
    
    Args:
        organ_overlaps: Dictionary of organ overlaps (organ_name: overlap_percentage)
        closest_organs: Dictionary of closest organs (organ_name: {'distance_voxels': float})
        target_count: Target number of total organs to return
        
    Returns:
        Dictionary with 'overlaps' and 'closest' keys containing the combined results
    """
    # Start with overlaps (highest priority)
    final_overlaps = {}
    final_closest = {}
    used_organs = set()
    
    # Add overlaps first (up to target_count)
    overlap_items = sorted(organ_overlaps.items(), key=lambda x: x[1], reverse=True)  # Sort by overlap percentage desc
    for organ_name, overlap_pct in overlap_items[:target_count]:
        final_overlaps[organ_name] = overlap_pct
        used_organs.add(organ_name)
    
    # Add closest organs to fill up to target_count (skip organs already in overlaps)
    remaining_slots = target_count - len(final_overlaps)
    if remaining_slots > 0:
        closest_items = sorted(closest_organs.items(), key=lambda x: x[1]['distance_voxels'])  # Sort by distance asc
        for organ_name, distance_info in closest_items:
            if organ_name not in used_organs and len(final_closest) < remaining_slots:
                final_closest[organ_name] = distance_info
                used_organs.add(organ_name)
    
    return {
        'overlaps': final_overlaps,
        'closest': final_closest
    }


def determine_vertebrae_level(lesion_voxels: np.ndarray, vertebrae_axial_ranges: Dict) -> Optional[List[str]]:
    """
    Determine vertebrae level based on lesion's axial position
    Uses actual min/max vertebrae coordinates for accurate matching
    
    Args:
        lesion_voxels: Boolean array indicating lesion locations
        vertebrae_axial_ranges: Pre-computed vertebrae axial ranges
        
    Returns:
        List of vertebrae names that the lesion overlaps with, or None if not in vertebral region
    """
    # Calculate lesion's axial extent
    lesion_coords = np.where(lesion_voxels)
    lesion_z_min = np.min(lesion_coords[2])  # Minimum Z coordinate (using Dim2)
    lesion_z_max = np.max(lesion_coords[2])  # Maximum Z coordinate (using Dim2)
    
    # Use pre-computed vertebrae axial ranges
    if not vertebrae_axial_ranges:
        return None
    
    # Find all vertebrae that overlap with the lesion
    best_overlap_ratio = 0.0
    best_vertebra_name = None
    
    for vertebrae_name, vertebra_info in vertebrae_axial_ranges.items():
        # Calculate overlap between lesion and vertebra in Z-axis
        overlap_min = max(lesion_z_min, vertebra_info['z_min'])
        overlap_max = min(lesion_z_max, vertebra_info['z_max'])
        
        if overlap_max >= overlap_min:  # There is some overlap
            overlap_span = overlap_max - overlap_min + 1
            vertebra_span = vertebra_info['z_max'] - vertebra_info['z_min'] + 1
            overlap_ratio = overlap_span / vertebra_span  # Ratio relative to vertebra size
            
            if overlap_ratio > best_overlap_ratio:
                best_overlap_ratio = overlap_ratio
                best_vertebra_name = vertebrae_name
                
    # Return the list of overlapping vertebrae
    return [best_vertebra_name] if best_vertebra_name else []


def precompute_nearest_organ_fields(organ_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute per-voxel:
    - nearest_organ_dist_voxels: distance (voxels) to nearest organ voxel (label>0)
    - nearest_organ_label: label of that nearest organ voxel
    
    Args:
        organ_data: Organ segmentation data
        
    Returns:
        Tuple of (distance_array, label_array)
    """
    # EDT measures distance from non-zero voxels to the nearest zero voxel.
    # So we set organ voxels to 0 (False) and everything else to 1 (True).
    edt_input = (organ_data == 0)
    dist_voxels, idx = ndi.distance_transform_edt(
        edt_input,
        return_indices=True
    )
    nearest_lab = organ_data[idx[0], idx[1], idx[2]]

    return dist_voxels.astype(np.float32), nearest_lab.astype(np.int32)


def find_closest_organs(lesion_voxels: np.ndarray, organ_data: np.ndarray, organ_mapping: Dict[int, str],
                       nearest_organ_dist_voxels: np.ndarray, nearest_organ_label: np.ndarray,
                       image_shape: Tuple[int, int, int], num_closest: int = 3,
                       max_voxels: float = 400.0, min_surface_frac: float = 0.0) -> Dict[str, Dict[str, any]]:
    """
    Nearest-organ query using a precomputed Euclidean distance field to organ surfaces.
    Returns up to num_closest organs with distance and spatial relationship information.
    
    Args:
        lesion_voxels: Boolean array indicating lesion locations
        organ_data: Organ segmentation data
        organ_mapping: Mapping from organ labels to names
        nearest_organ_dist_voxels: Pre-computed distance field
        nearest_organ_label: Pre-computed label field
        image_shape: Shape of the image
        num_closest: Number of closest organs to return
        max_voxels: Maximum distance in voxels
        min_surface_frac: Minimum surface fraction
        
    Returns:
        Dictionary of closest organs with distance and spatial information
    """
    # Work on lesion surface for geometric correctness and speed
    surf = lesion_surface(lesion_voxels.astype(bool))
    if not np.any(surf):
        return {}

    # Calculate lesion center for spatial relationship analysis
    lesion_coords = np.where(lesion_voxels)
    lesion_center = np.array([
        np.mean(lesion_coords[0]),  # x (dim0)
        np.mean(lesion_coords[1]),  # y (dim1) 
        np.mean(lesion_coords[2])   # z (dim2)
    ])

    labs = nearest_organ_label[surf]
    dvox = nearest_organ_dist_voxels[surf]

    # Ignore background
    valid = labs > 0
    labs, dvox = labs[valid], dvox[valid]
    if labs.size == 0:
        return {}

    # Aggregate per organ: minimal distance, plus optional surface support
    n_surf = int(surf.sum())
    per_org = {}
    for lab in np.unique(labs):
        sel = (labs == lab)
        di  = float(dvox[sel].min())
        frac = float((dvox[sel] <= max_voxels).sum()) / float(n_surf)
        if (di <= max_voxels) or (frac >= min_surface_frac):
            name = organ_mapping.get(int(lab), f"label_{int(lab)}")
            
            # Calculate spatial relationship
            organ_mask = (organ_data == lab)
            spatial_relationship = calculate_spatial_relationship(lesion_center, organ_mask, image_shape)
            
            per_org[name] = {
                "distance_voxels": di, 
                "surface_frac": frac,
                "spatial_relationship": spatial_relationship
            }

    # Rank by true distance, then by supporting surface fraction
    ranked = sorted(per_org.items(),
                    key=lambda kv: (kv[1]["distance_voxels"], -kv[1]["surface_frac"]))
    
    # Return distance and spatial relationship info
    return {k: {
        "distance_voxels": v["distance_voxels"],
        "spatial_relationship": v["spatial_relationship"]
    } for k, v in ranked[:num_closest]}


def save_lesion_mask(final_lesion_mask: np.ndarray, output_path: str, 
                    suv_img_affine: np.ndarray, suv_img_header) -> None:
    """
    Save the final lesion segmentation mask as a NIfTI file
    
    Args:
        final_lesion_mask: Final lesion mask array
        output_path: Path where to save the lesion mask (should end with .nii or .nii.gz)
        suv_img_affine: Affine matrix from SUV image
        suv_img_header: Header from SUV image
    """
    # Create NIfTI image using the same header and affine as the original SUV image
    lesion_img = nib.Nifti1Image(
        final_lesion_mask.astype(np.int16), 
        suv_img_affine, 
        suv_img_header
    )
    
    # Save the mask
    nib.save(lesion_img, output_path)
    print(f"Lesion segmentation mask saved to: {output_path}")


def precompute_vertebrae_levels(organ_data: np.ndarray, organ_mapping: Dict[int, str]) -> Tuple[List[Tuple[int, str]], Dict[str, Dict]]:
    """
    Pre-calculate vertebrae axial coordinate ranges for lesion matching
    
    Args:
        organ_data: Organ segmentation data
        organ_mapping: Mapping from organ labels to names
        
    Returns:
        Tuple of (vertebrae_order, vertebrae_axial_ranges)
    """
    # Define vertebrae in anatomical order (inferior to superior)
    vertebrae_order = [
        (25, "sacrum"),
        (26, "vertebrae_S1"),
        (27, "vertebrae_L5"),
        (28, "vertebrae_L4"), 
        (29, "vertebrae_L3"),
        (30, "vertebrae_L2"),
        (31, "vertebrae_L1"),
        (32, "vertebrae_T12"),
        (33, "vertebrae_T11"),
        (34, "vertebrae_T10"),
        (35, "vertebrae_T9"),
        (36, "vertebrae_T8"),
        (37, "vertebrae_T7"),
        (38, "vertebrae_T6"),
        (39, "vertebrae_T5"),
        (40, "vertebrae_T4"),
        (41, "vertebrae_T3"),
        (42, "vertebrae_T2"),
        (43, "vertebrae_T1"),
        (44, "vertebrae_C7"),
        (45, "vertebrae_C6"),
        (46, "vertebrae_C5"),
        (47, "vertebrae_C4"),
        (48, "vertebrae_C3"),
        (49, "vertebrae_C2"),
        (50, "vertebrae_C1")
    ]
    
    # Calculate actual min/max axial coordinates for each vertebra
    vertebrae_axial_ranges = {}
    
    for label, name in vertebrae_order:
        if label in organ_mapping:
            # Find vertebra in organ mask
            vertebra_mask = (organ_data == label)
            vertebra_voxel_count = np.sum(vertebra_mask)
            
            if np.any(vertebra_mask):
                # Get all coordinates for this vertebra
                vertebra_coords = np.where(vertebra_mask)
                
                # Calculate spans in all three dimensions to determine correct axis
                dim0_min = np.min(vertebra_coords[0])  # First dimension
                dim0_max = np.max(vertebra_coords[0])
                dim0_span = dim0_max - dim0_min + 1
                
                dim1_min = np.min(vertebra_coords[1])  # Second dimension
                dim1_max = np.max(vertebra_coords[1])
                dim1_span = dim1_max - dim1_min + 1
                
                dim2_min = np.min(vertebra_coords[2])  # Third dimension
                dim2_max = np.max(vertebra_coords[2])
                dim2_span = dim2_max - dim2_min + 1
                
                # Use Dim2 as the axial direction (head-to-foot)
                z_min = dim2_min
                z_max = dim2_max
                z_center = np.mean(vertebra_coords[2])  # Center along Dim2
                z_span = z_max - z_min + 1
                
                vertebrae_axial_ranges[name] = {
                    'z_min': z_min,
                    'z_max': z_max,
                    'z_center': z_center,
                    'label': label,
                    'voxel_count': vertebra_voxel_count
                }
            else:
                pass  # Vertebra not found in organ data
        else:
            pass  # Vertebra not in organ mapping
    
    # Sort vertebrae by their center position for display
    sorted_vertebrae = sorted(vertebrae_axial_ranges.items(), key=lambda x: x[1]['z_center'])
    
    return vertebrae_order, vertebrae_axial_ranges
