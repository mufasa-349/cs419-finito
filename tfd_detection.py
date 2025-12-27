"""
Temporal Frame Differencing (TFD) + CHT Verification + Temporal CHT Signature

Based on proposal: "Mitosis Detection by Temporal Frame Differencing"

This module implements:
1. Temporal Frame Differencing: Dt(x, y) = |It(x, y) - It-1(x, y)|
2. Morphological operations (opening and closing) to remove noise
3. Candidate region detection from significant temporal changes
4. Symmetry-Based Morphological Verification using Circle Hough Transform (CHT)
5. Temporal CHT Signature: Track CHT scores across frames and detect rise-peak-drop pattern
"""

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import json

# Try to import tifffile, fallback to PIL if not available
try:
    import tifffile
    USE_TIFFFILE = True
except ImportError:
    try:
        from PIL import Image
        USE_TIFFFILE = False
        print("tifffile not found, using PIL instead")
    except ImportError:
        raise ImportError("Neither tifffile nor PIL is available. Please install one: pip install tifffile or pip install Pillow")


def load_image(image_path):
    """Load a single grayscale image."""
    if USE_TIFFFILE:
        img = tifffile.imread(image_path)
    else:
        img = np.array(Image.open(image_path))
    
    # Ensure grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    return img


def preprocess_image(image):
    """
    Preprocess image: Normalize intensity to [0, 255]
    
    Args:
        image: Input grayscale image
        
    Returns:
        normalized: Intensity-normalized image
    """
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        normalized = image.astype(np.uint8)
    
    return normalized


def temporal_frame_differencing(frame_t, frame_t_minus_1):
    """
    Compute Temporal Frame Differencing.
    
    Dt(x, y) = |It(x, y) - It-1(x, y)|
    
    Args:
        frame_t: Current frame It
        frame_t_minus_1: Previous frame It-1
        
    Returns:
        difference_map: Pixel-wise absolute difference Dt
    """
    # Ensure both frames have same shape
    if frame_t.shape != frame_t_minus_1.shape:
        raise ValueError(f"Frame shapes don't match: {frame_t.shape} vs {frame_t_minus_1.shape}")
    
    # Compute absolute difference
    difference_map = cv2.absdiff(frame_t, frame_t_minus_1)
    
    return difference_map


def apply_morphological_operations(difference_map, kernel_size=5):
    """
    Apply morphological opening and closing to remove noise.
    
    Args:
        difference_map: Temporal difference map Dt
        kernel_size: Size of morphological kernel
        
    Returns:
        cleaned_map: Morphologically cleaned difference map
    """
    # Create morphological kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Opening: removes small noise (erosion followed by dilation)
    opened = cv2.morphologyEx(difference_map, cv2.MORPH_OPEN, kernel)
    
    # Closing: fills small holes (dilation followed by erosion)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    return closed


def detect_candidate_regions(cleaned_map, min_change_threshold=30, min_area=50):
    """
    Identify regions with significant temporal change as candidate mitotic areas.
    
    Args:
        cleaned_map: Morphologically cleaned difference map
        min_change_threshold: Minimum intensity change to consider (0-255)
        min_area: Minimum area for a candidate region (pixels)
        
    Returns:
        candidates: List of candidate regions with properties
        binary_mask: Binary mask of candidate regions
    """
    # Threshold to get significant changes
    _, binary_mask = cv2.threshold(cleaned_map, min_change_threshold, 255, cv2.THRESH_BINARY)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    
    candidates = []
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Filter by minimum area
        if area >= min_area:
            # Get bounding box
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Calculate centroid
            cx = int(centroids[i, 0])
            cy = int(centroids[i, 1])
            
            # Calculate mean change score for this region
            region_mask = (labels == i).astype(np.uint8)
            mean_change_score = np.mean(cleaned_map[region_mask > 0])
            
            candidates.append({
                'id': i,
                'bbox': (x, y, x + w, y + h),
                'centroid': (cx, cy),
                'area': int(area),
                'mean_change_score': float(mean_change_score),
                'mask': region_mask * 255
            })
    
    return candidates, binary_mask


def compute_cht_symmetry_score(candidate_region, image, min_radius=10, max_radius=50):
    """
    Compute circular symmetry score using Circle Hough Transform (CHT).
    
    Cells entering mitosis tend to round up, leading to a high CHT response.
    This helps reject false positives from TFD caused by background or illumination changes.
    
    Args:
        candidate_region: Candidate region dictionary with 'bbox' and 'mask'
        image: Original grayscale image (frame t)
        min_radius: Minimum circle radius for CHT
        max_radius: Maximum circle radius for CHT
        
    Returns:
        symmetry_score: CHT response score (0-1, higher = more circular)
        best_circle: Best matching circle (center, radius) or None
    """
    x, y, x2, y2 = candidate_region['bbox']
    
    # Extract region of interest with padding
    padding = max_radius + 10
    y1 = max(0, y - padding)
    y2_roi = min(image.shape[0], y2 + padding)
    x1 = max(0, x - padding)
    x2_roi = min(image.shape[1], x2 + padding)
    
    roi = image[y1:y2_roi, x1:x2_roi]
    
    if roi.size == 0:
        return 0.0, None
    
    # Apply Circle Hough Transform
    # Use more lenient parameters for phase-contrast images
    circles = cv2.HoughCircles(
        roi,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=max(10, max_radius),  # Allow closer circles
        param1=30,   # Lower threshold for edge detection (more sensitive)
        param2=15,   # Lower accumulator threshold (more circles detected)
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    if circles is None:
        return 0.0, None
    
    circles = np.round(circles[0, :]).astype("int")
    
    # Get candidate mask in ROI coordinates
    candidate_mask_roi = candidate_region['mask'][y1:y2_roi, x1:x2_roi]
    candidate_area = np.sum(candidate_mask_roi > 0)
    
    if candidate_area == 0:
        return 0.0, None
    
    # Find the circle with best overlap with candidate region
    best_score = 0.0
    best_circle = None
    
    for (cx, cy, r) in circles:
        # Create circle mask
        circle_mask = np.zeros_like(roi)
        cv2.circle(circle_mask, (cx, cy), r, 255, -1)
        
        # Calculate overlap with candidate region
        overlap = cv2.bitwise_and(candidate_mask_roi, circle_mask)
        overlap_area = np.sum(overlap > 0)
        
        # Symmetry score: overlap ratio
        overlap_ratio = overlap_area / candidate_area if candidate_area > 0 else 0
        
        # Also consider how well the circle fits (area ratio)
        circle_area = np.pi * r * r
        area_ratio = min(candidate_area, circle_area) / max(candidate_area, circle_area) if max(candidate_area, circle_area) > 0 else 0
        
        # Combined score: overlap * area_fit
        combined_score = overlap_ratio * area_ratio
        
        if combined_score > best_score:
            best_score = combined_score
            # Adjust circle center to original image coordinates
            best_circle = {
                'center': (cx + x1, cy + y1),
                'radius': r,
                'overlap_ratio': overlap_ratio,
                'area_ratio': area_ratio
            }
    
    return best_score, best_circle


def verify_candidates_with_cht(candidates, image, min_radius=10, max_radius=50, 
                                symmetry_threshold=0.3):
    """
    Verify TFD candidates using Circle Hough Transform.
    
    Filters candidates based on circular symmetry. Cells entering mitosis
    tend to round up, leading to a high CHT response.
    
    Args:
        candidates: List of candidate regions from TFD
        image: Original grayscale image (frame t)
        min_radius: Minimum circle radius for CHT
        max_radius: Maximum circle radius for CHT
        symmetry_threshold: Minimum symmetry score to keep a candidate
        
    Returns:
        verified_candidates: List of verified candidates with CHT scores
        rejected_candidates: List of rejected candidates
    """
    verified_candidates = []
    rejected_candidates = []
    
    for candidate in candidates:
        # Compute CHT symmetry score
        symmetry_score, best_circle = compute_cht_symmetry_score(
            candidate, image, min_radius, max_radius
        )
        
        # Add symmetry information to candidate
        candidate['symmetry_score'] = symmetry_score
        candidate['cht_circle'] = best_circle
        
        # Keep candidate if symmetry score is above threshold
        if symmetry_score >= symmetry_threshold:
            verified_candidates.append(candidate)
        else:
            rejected_candidates.append(candidate)
    
    return verified_candidates, rejected_candidates


def compute_distance(point1, point2):
    """Compute Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def track_candidates_across_frames(all_results, max_distance=50):
    """
    Track verified candidates across frames using spatial proximity.
    
    Args:
        all_results: List of results from all frame pairs
        max_distance: Maximum distance to consider same candidate (pixels)
        
    Returns:
        tracks: Dictionary mapping track_id to list of (frame_idx, candidate) tuples
    """
    tracks = {}
    next_track_id = 1
    
    for frame_idx, result in enumerate(all_results):
        verified_candidates = [c for c in result['candidates'] if c.get('verified', False)]
        
        for candidate in verified_candidates:
            centroid = candidate['centroid']
            
            # Find closest existing track
            best_track_id = None
            best_distance = max_distance
            
            for track_id, track_points in tracks.items():
                # Get last point in track
                last_frame_idx, last_candidate = track_points[-1]
                
                # Only consider recent frames (within 3 frames)
                if frame_idx - last_frame_idx <= 3:
                    distance = compute_distance(centroid, last_candidate['centroid'])
                    if distance < best_distance:
                        best_distance = distance
                        best_track_id = track_id
            
            # Assign to existing track or create new track
            if best_track_id is not None:
                tracks[best_track_id].append((frame_idx, candidate))
            else:
                tracks[next_track_id] = [(frame_idx, candidate)]
                next_track_id += 1
    
    return tracks


def compute_temporal_cht_signature(track, all_results, directory, min_radius=10, max_radius=50):
    """
    Compute Temporal CHT Signature for a tracked candidate.
    
    Tracks CHT score across consecutive frames, forming a temporal feature vector.
    
    Args:
        track: List of (frame_idx, candidate) tuples
        all_results: List of all frame pair results
        directory: Directory containing images
        min_radius: Minimum circle radius for CHT
        max_radius: Maximum circle radius for CHT
        
    Returns:
        signature: List of CHT scores across frames
        frame_indices: List of frame indices
    """
    signature = []
    frame_indices = []
    
    for frame_idx, candidate in track:
        # Get the frame for this candidate
        result = all_results[frame_idx]
        frame_name = result['frame_t']
        
        # Load frame to compute CHT
        frame_path = Path(directory) / frame_name
        if not frame_path.exists():
            continue
        
        frame = load_image(str(frame_path))
        frame = preprocess_image(frame)
        
        # Recreate mask from bbox if not available
        if 'mask' not in candidate:
            x, y, x2, y2 = candidate['bbox']
            mask = np.zeros(frame.shape, dtype=np.uint8)
            mask[y:y2, x:x2] = 255
            candidate_with_mask = candidate.copy()
            candidate_with_mask['mask'] = mask
        else:
            candidate_with_mask = candidate
        
        # Compute CHT score for this candidate at this frame
        symmetry_score, _ = compute_cht_symmetry_score(
            candidate_with_mask, frame, min_radius, max_radius
        )
        
        signature.append(symmetry_score)
        frame_indices.append(frame_idx)
    
    return signature, frame_indices


def detect_rise_peak_drop_pattern(signature, min_rise=0.05, min_peak_height=0.15, min_drop=0.05):
    """
    Detect rise-peak-drop pattern in temporal CHT signature.
    
    True mitosis follows a consistent pattern:
    - Rise: CHT score increases (cell rounds up)
    - Peak: Maximum CHT score (maximum circularity during mitosis)
    - Drop: CHT score decreases (cell division completes)
    
    Args:
        signature: List of CHT scores
        min_rise: Minimum increase from start to peak
        min_peak_height: Minimum peak value
        min_drop: Minimum decrease from peak to end
        
    Returns:
        has_pattern: Boolean indicating if pattern is detected
        peak_idx: Index of peak in signature
        pattern_info: Dictionary with pattern details
    """
    if len(signature) < 3:
        return False, None, {}
    
    signature = np.array(signature)
    
    # Find peak
    peak_idx = np.argmax(signature)
    peak_value = signature[peak_idx]
    
    # Check if peak is high enough
    if peak_value < min_peak_height:
        return False, None, {}
    
    # Check rise phase (before peak)
    if peak_idx > 0:
        start_value = signature[0]
        rise = peak_value - start_value
        if rise < min_rise:
            return False, None, {}
    else:
        rise = 0
    
    # Check drop phase (after peak)
    if peak_idx < len(signature) - 1:
        end_value = signature[-1]
        drop = peak_value - end_value
        if drop < min_drop:
            return False, None, {}
    else:
        drop = 0
    
    # Pattern detected
    pattern_info = {
        'peak_idx': int(peak_idx),
        'peak_value': float(peak_value),
        'start_value': float(signature[0]),
        'end_value': float(signature[-1]),
        'rise': float(rise),
        'drop': float(drop),
        'signature_length': len(signature)
    }
    
    return True, peak_idx, pattern_info


def analyze_temporal_signatures(all_results, directory, max_distance=50, 
                                min_rise=0.05, min_peak_height=0.15, min_drop=0.05,
                                min_radius=10, max_radius=50):
    """
    Analyze temporal CHT signatures for all tracked candidates.
    
    Args:
        all_results: List of all frame pair results
        directory: Directory containing images
        max_distance: Maximum distance for candidate tracking
        min_rise: Minimum rise for pattern detection
        min_peak_height: Minimum peak height
        min_drop: Minimum drop for pattern detection
        min_radius: Minimum circle radius for CHT
        max_radius: Maximum circle radius for CHT
        
    Returns:
        mitotic_events: List of detected mitotic events
        all_tracks: All candidate tracks with signatures
    """
    # Track candidates across frames
    tracks = track_candidates_across_frames(all_results, max_distance=max_distance)
    
    mitotic_events = []
    all_tracks = []
    
    for track_id, track_points in tracks.items():
        if len(track_points) < 3:  # Need at least 3 frames for pattern
            continue
        
        # Compute temporal signature
        signature, frame_indices = compute_temporal_cht_signature(
            track_points, all_results, directory, min_radius, max_radius
        )
        
        if len(signature) < 3:
            continue
        
        # Detect rise-peak-drop pattern
        has_pattern, peak_idx, pattern_info = detect_rise_peak_drop_pattern(
            signature, min_rise, min_peak_height, min_drop
        )
        
        # Get candidate at peak
        peak_frame_idx = frame_indices[peak_idx] if peak_idx is not None else None
        peak_candidate = track_points[peak_idx][1] if peak_idx is not None else None
        
        track_info = {
            'track_id': track_id,
            'length': len(track_points),
            'frame_indices': frame_indices,
            'signature': signature,
            'has_mitotic_pattern': has_pattern,
            'peak_frame_idx': peak_frame_idx,
            'pattern_info': pattern_info,
            'peak_candidate': peak_candidate
        }
        
        all_tracks.append(track_info)
        
        if has_pattern:
            mitotic_events.append(track_info)
    
    return mitotic_events, all_tracks


def visualize_tfd_results(frame_t, frame_t_minus_1, difference_map, cleaned_map, 
                         candidates, verified_candidates=None, output_path=None):
    """
    Visualize TFD results.
    
    Args:
        frame_t: Current frame
        frame_t_minus_1: Previous frame
        difference_map: Raw difference map
        cleaned_map: Morphologically cleaned difference map
        candidates: List of candidate regions
        output_path: Path to save visualization
    """
    # Create visualization with 4 panels
    h, w = frame_t.shape
    
    # Resize if needed for better visualization
    vis_h, vis_w = h, w * 4
    
    vis_image = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
    
    # Panel 1: Frame t-1
    vis_image[:, :w] = cv2.cvtColor(frame_t_minus_1, cv2.COLOR_GRAY2BGR)
    cv2.putText(vis_image, "Frame t-1", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Panel 2: Frame t
    vis_image[:, w:2*w] = cv2.cvtColor(frame_t, cv2.COLOR_GRAY2BGR)
    cv2.putText(vis_image, "Frame t", (w + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Panel 3: Difference map (colormap)
    diff_colored = cv2.applyColorMap(difference_map, cv2.COLORMAP_JET)
    vis_image[:, 2*w:3*w] = diff_colored
    cv2.putText(vis_image, "Difference Map", (2*w + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Panel 4: Candidates on cleaned map
    cleaned_colored = cv2.applyColorMap(cleaned_map, cv2.COLORMAP_JET)
    vis_image[:, 3*w:4*w] = cleaned_colored
    
    # Draw candidate regions
    # Draw all candidates in blue (unverified)
    for candidate in candidates:
        x, y, x2, y2 = candidate['bbox']
        x_vis = x + 3*w
        x2_vis = x2 + 3*w
        
        # Check if this candidate is verified
        is_verified = verified_candidates is not None and any(
            c['id'] == candidate['id'] for c in verified_candidates
        )
        
        if is_verified:
            # Verified candidates: green
            cv2.rectangle(vis_image, (x_vis, y), (x2_vis, y2), (0, 255, 0), 2)
        else:
            # Unverified candidates: blue (semi-transparent)
            overlay = vis_image.copy()
            cv2.rectangle(overlay, (x_vis, y), (x2_vis, y2), (255, 0, 0), 2)
            vis_image = cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0)
        
        # Draw centroid
        cx, cy = candidate['centroid']
        cv2.circle(vis_image, (cx + 3*w, cy), 5, (0, 0, 255), -1)
        
        # Draw ID and symmetry score if available
        label = f"ID{candidate['id']}"
        if 'symmetry_score' in candidate:
            label += f" S:{candidate['symmetry_score']:.2f}"
        
        cv2.putText(vis_image, label, 
                   (cx + 3*w + 8, cy - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(vis_image, label, 
                   (cx + 3*w + 8, cy - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw CHT circle if available
        if verified_candidates and candidate.get('cht_circle'):
            circle = candidate['cht_circle']
            center = circle['center']
            radius = circle['radius']
            # Adjust center for panel 4
            center_vis = (center[0] + 3*w, center[1])
            cv2.circle(vis_image, center_vis, radius, (0, 255, 255), 2)  # Yellow circle
    
    label_text = "Candidates (Green=verified, Blue=rejected, Red=centroid, Yellow=CHT circle)"
    cv2.putText(vis_image, label_text, (3*w + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if output_path:
        cv2.imwrite(str(output_path), vis_image)
    
    return vis_image


def process_tfd_pair(frame_t_path, frame_t_minus_1_path, 
                    min_change_threshold=30, min_area=50,
                    kernel_size=5, 
                    min_radius=10, max_radius=50, symmetry_threshold=0.3,
                    visualize=True, output_dir=None):
    """
    Process a pair of frames with TFD.
    
    Args:
        frame_t_path: Path to current frame
        frame_t_minus_1_path: Path to previous frame
        min_change_threshold: Minimum change threshold
        min_area: Minimum area for candidates
        kernel_size: Morphological kernel size
        visualize: Whether to create visualization
        output_dir: Output directory for results
        
    Returns:
        result: Dictionary with TFD results
    """
    # Load frames
    frame_t = load_image(frame_t_path)
    frame_t_minus_1 = load_image(frame_t_minus_1_path)
    
    # Preprocess
    frame_t = preprocess_image(frame_t)
    frame_t_minus_1 = preprocess_image(frame_t_minus_1)
    
    # Step 1: Temporal Frame Differencing
    difference_map = temporal_frame_differencing(frame_t, frame_t_minus_1)
    
    # Step 2: Morphological operations
    cleaned_map = apply_morphological_operations(difference_map, kernel_size=kernel_size)
    
    # Step 3: Detect candidate regions
    candidates, binary_mask = detect_candidate_regions(
        cleaned_map, 
        min_change_threshold=min_change_threshold,
        min_area=min_area
    )
    
    # Step 4: Verify candidates with CHT
    verified_candidates, rejected_candidates = verify_candidates_with_cht(
        candidates, frame_t,
        min_radius=min_radius,
        max_radius=max_radius,
        symmetry_threshold=symmetry_threshold
    )
    
    # Visualization
    vis_image = None
    if visualize:
        output_path = None
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            frame_name = Path(frame_t_path).stem
            output_path = output_dir / f"{frame_name}_tfd.png"
        
        vis_image = visualize_tfd_results(
            frame_t, frame_t_minus_1, difference_map, cleaned_map,
            candidates, verified_candidates=verified_candidates,
            output_path=output_path
        )
    
    # Prepare result
    result = {
        'frame_t': Path(frame_t_path).name,
        'frame_t_minus_1': Path(frame_t_minus_1_path).name,
        'num_candidates': len(candidates),
        'num_verified': len(verified_candidates),
        'num_rejected': len(rejected_candidates),
        'candidates': [
            {
                'id': int(c['id']),
                'bbox': (int(c['bbox'][0]), int(c['bbox'][1]), 
                        int(c['bbox'][2]), int(c['bbox'][3])),
                'centroid': (int(c['centroid'][0]), int(c['centroid'][1])),
                'area': int(c['area']),
                'mean_change_score': float(c['mean_change_score']),
                'symmetry_score': float(c.get('symmetry_score', 0.0)),
                'verified': c['id'] in [vc['id'] for vc in verified_candidates],
                'cht_circle': {
                    'center': (int(c['cht_circle']['center'][0]), int(c['cht_circle']['center'][1])),
                    'radius': int(c['cht_circle']['radius']),
                    'overlap_ratio': float(c['cht_circle']['overlap_ratio']),
                    'area_ratio': float(c['cht_circle']['area_ratio'])
                } if c.get('cht_circle') else None
            }
            for c in candidates
        ]
    }
    
    return result


def process_sequence(directory, output_dir=None, 
                    min_change_threshold=30, min_area=50,
                    kernel_size=5,
                    min_radius=10, max_radius=50, symmetry_threshold=0.3,
                    analyze_temporal=True, max_tracking_distance=50,
                    min_rise=0.05, min_peak_height=0.15, min_drop=0.05):
    """
    Process a sequence of images with TFD.
    
    Args:
        directory: Directory containing images
        output_dir: Output directory for results
        min_change_threshold: Minimum change threshold
        min_area: Minimum area for candidates
        kernel_size: Morphological kernel size
    """
    directory = Path(directory)
    image_files = sorted(directory.glob("*.tif"))
    
    if len(image_files) < 2:
        print("Need at least 2 images for TFD")
        return
    
    print(f"Processing {len(image_files) - 1} frame pairs...")
    
    all_results = []
    
    for i in tqdm(range(1, len(image_files))):
        frame_t_path = image_files[i]
        frame_t_minus_1_path = image_files[i-1]
        
        result = process_tfd_pair(
            str(frame_t_path),
            str(frame_t_minus_1_path),
            min_change_threshold=min_change_threshold,
            min_area=min_area,
            kernel_size=kernel_size,
            min_radius=min_radius,
            max_radius=max_radius,
            symmetry_threshold=symmetry_threshold,
            visualize=True,
            output_dir=output_dir
        )
        all_results.append(result)
    
    # Save results to JSON
    if output_dir:
        output_dir = Path(output_dir)
        results_path = output_dir / "tfd_results.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # Analyze temporal signatures if requested
    mitotic_events = []
    all_tracks = []
    if analyze_temporal:
        print("\nAnalyzing temporal CHT signatures...")
        mitotic_events, all_tracks = analyze_temporal_signatures(
            all_results,
            directory,
            max_distance=max_tracking_distance,
            min_rise=min_rise,
            min_peak_height=min_peak_height,
            min_drop=min_drop,
            min_radius=min_radius,
            max_radius=max_radius
        )
        
        # Save temporal analysis results
        if output_dir:
            output_dir = Path(output_dir)
            temporal_path = output_dir / "temporal_analysis.json"
            with open(temporal_path, 'w') as f:
                json.dump({
                    'mitotic_events': [
                        {
                            'track_id': e['track_id'],
                            'length': e['length'],
                            'frame_indices': e['frame_indices'],
                            'signature': e['signature'],
                            'peak_frame_idx': e['peak_frame_idx'],
                            'pattern_info': e['pattern_info']
                        }
                        for e in mitotic_events
                    ],
                    'all_tracks': [
                        {
                            'track_id': t['track_id'],
                            'length': t['length'],
                            'has_mitotic_pattern': t['has_mitotic_pattern'],
                            'signature': t['signature']
                        }
                        for t in all_tracks
                    ]
                }, f, indent=2)
    
    # Print summary
    total_candidates = sum(r['num_candidates'] for r in all_results)
    total_verified = sum(r['num_verified'] for r in all_results)
    total_rejected = sum(r['num_rejected'] for r in all_results)
    avg_candidates = total_candidates / len(all_results) if all_results else 0
    avg_verified = total_verified / len(all_results) if all_results else 0
    
    print(f"\nTFD + CHT + Temporal Signature Summary:")
    print(f"Total frame pairs processed: {len(all_results)}")
    print(f"Total candidates detected: {total_candidates}")
    print(f"Total verified (CHT): {total_verified}")
    print(f"Total rejected (CHT): {total_rejected}")
    print(f"Average candidates per pair: {avg_candidates:.2f}")
    print(f"Average verified per pair: {avg_verified:.2f}")
    print(f"Verification rate: {100*total_verified/total_candidates:.1f}%" if total_candidates > 0 else "N/A")
    
    if analyze_temporal:
        print(f"\nTemporal Analysis:")
        print(f"Total tracks: {len(all_tracks)}")
        print(f"Mitotic events detected (rise-peak-drop pattern): {len(mitotic_events)}")
        if len(all_tracks) > 0:
            print(f"Mitosis detection rate: {100*len(mitotic_events)/len(all_tracks):.1f}%")
    
    return all_results


def main():
    """Main function to test TFD."""
    # Configuration
    input_dir = "hela_dataset-train-01"
    output_dir = "tfd_results"
    
    # TFD parameters
    min_change_threshold = 30  # Minimum intensity change (0-255)
    min_area = 50              # Minimum area for candidate regions (pixels)
    kernel_size = 5           # Morphological kernel size
    
    # CHT parameters
    min_radius = 10           # Minimum circle radius for CHT
    max_radius = 50           # Maximum circle radius for CHT
    symmetry_threshold = 0.1  # Minimum symmetry score to verify candidate (lowered for phase-contrast)
    
    # Temporal CHT Signature parameters
    analyze_temporal = True   # Whether to analyze temporal signatures
    max_tracking_distance = 50  # Maximum distance for candidate tracking (pixels)
    min_rise = 0.05          # Minimum rise for pattern detection
    min_peak_height = 0.15   # Minimum peak height for pattern detection
    min_drop = 0.05          # Minimum drop for pattern detection
    
    # Process sequence
    results = process_sequence(
        input_dir,
        output_dir=output_dir,
        min_change_threshold=min_change_threshold,
        min_area=min_area,
        kernel_size=kernel_size,
        min_radius=min_radius,
        max_radius=max_radius,
        symmetry_threshold=symmetry_threshold,
        analyze_temporal=analyze_temporal,
        max_tracking_distance=max_tracking_distance,
        min_rise=min_rise,
        min_peak_height=min_peak_height,
        min_drop=min_drop
    )
    
    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()

