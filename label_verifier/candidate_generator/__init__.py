"""
Candidate Generator Package for Label Verification

This package provides a clean, modular approach to generating candidate bounding boxes
for label verification. The pipeline is split into numbered steps for easy reading,
modification, and testing.

Public API:
- generate_candidates(page_img, ref_icon, config, debug_dir=None)
- match_page_candidates(page_img, icon_tuples, config, debug_dir=None)

Configuration Parameters:
- dbscan_eps: DBSCAN clustering epsilon (default: 5.0)
- dbscan_min_samples: DBSCAN minimum samples (default: 8)
- dbscan_min_area: minimum area for boxes (default: 16)
- sift_point_radius: radius for rasterizing match points (default: 6)
- edge_low_thresh: Canny low threshold (default: 50)
- edge_high_thresh: Canny high threshold (default: 150)  
- edge_dilate_iter: edge dilation iterations (default: 1)
- edge_alt_low_thresh: alternative Canny low (default: 30)
- edge_alt_high_thresh: alternative Canny high (default: 100)
- edge_alt_dilate_iter: alternative dilation (default: 1)
- edge_border_density_thresh: border detection density (default: 0.02)
- edge_border_min_area_frac: border detection area fraction (default: 0.25)
- edge_coverage_threshold: edge-match coverage required (default: 0.25)
- merge_iou: IoU threshold for merging (default: 0.05)
- merge_gap: gap threshold for merging (default: 6)
- final_min_match_points: minimum match points in final boxes (default: 4)
- store_debug_images: whether to save debug images (default: False)
- debug_image_dir: directory for debug images
"""
import os
from . import feature_matching as match
from . import edge_detection as edges
from . import box_utils as boxes
from . import debug_utils as debug


# `generate_candidates` removed: callers should use `match_icon_candidates` directly.


def match_icon_candidates(label_img, ref_icon, edge_boxes_unused, config, debug_dir=None):
    """
    Generate candidates for a single icon with full numbered pipeline.
    
    Args:
        label_img: BGR page image
        ref_icon: BGR icon image  
        edge_boxes_unused: ignored (for compatibility)
        config: configuration dict
        debug_dir: optional debug output directory
        
    Returns:
        (kept_boxes, debug_dict): final boxes and debug information
    """
    dbg = {}
    if label_img is None or ref_icon is None:
        return [], dbg

    # STEP 1: Match point candidates (feature matching + clustering)
    try:
        match_boxes_raw, match_debug = match.match_point_candidates(
            label_img, ref_icon, config, debug_dir
        )
    except Exception:
        match_boxes_raw, match_debug = [], {}
    
    dbg.update(match_debug or {})
    
    # STEP 2: Compute edge-connected boxes
    edge_boxes_raw = edges.compute_edge_boxes(label_img, config)
    
    # Save debug overlay of all edge boxes
    _save_all_edge_boxes_debug(label_img, edge_boxes_raw, ref_icon, debug_dir, dbg)

    # Extract match points for later use
    match_points = dbg.get('match_points', [])
    
    # Read configuration parameters
    merge_iou_cfg = float(config.get('merge_iou', 0.05))
    merge_gap_cfg = int(config.get('merge_gap', 6))
    min_pts_cfg = int(config.get('final_min_match_points', 4))
    edge_coverage_threshold = float(config.get('edge_coverage_threshold', 0.1))

    # STEP 3: Simplify match boxes (merge by IoU)
    try:
        match_boxes = boxes.merge_boxes_simple(match_boxes_raw or [], iou_thresh=merge_iou_cfg)
    except Exception:
        match_boxes = match_boxes_raw or []

    # STEP 4: First-pass edge filter (coverage threshold)
    try:
        filtered_edge_boxes = []
        for edge_box in (edge_boxes_raw or []):
            edge_area = max(0, edge_box[2] * edge_box[3])
            if edge_area == 0:
                continue
            
            # Calculate coverage by match-cluster boxes
            sum_intersection = 0
            for match_box in (match_boxes_raw or []):
                try:
                    sum_intersection += boxes.intersection_area(edge_box, match_box)
                except Exception:
                    continue
            
            # Keep edge box if coverage >= threshold
            if float(sum_intersection) >= edge_coverage_threshold * float(edge_area):
                filtered_edge_boxes.append(edge_box)
                
        edge_boxes_filtered = filtered_edge_boxes
    except Exception:
        edge_boxes_filtered = edge_boxes_raw or []

    # STEP 5: Valid edge boxes (overlap with match boxes)  
    try:
        valid_edge_boxes = []
        if edge_boxes_filtered and match_boxes:
            for edge_box in edge_boxes_filtered:
                for match_box in match_boxes:
                    if boxes.iou(edge_box, match_box) > 0.0:
                        valid_edge_boxes.append(edge_box)
                        break
        else:
            valid_edge_boxes = []
    except Exception:
        valid_edge_boxes = []
        
    # Save debug overlays for filtered boxes
    _save_edge_filter_debug(label_img, edge_boxes_filtered, valid_edge_boxes, 
                           match_boxes_raw, ref_icon, debug_dir, dbg, merge_gap_cfg)

    # STEP 6: Pool and deduplicate (match + valid edge boxes)
    try:
        pool = []
        if match_boxes:
            pool.extend(match_boxes)
        if valid_edge_boxes:
            pool.extend(valid_edge_boxes)

        # Deduplicate and normalize
        seen = set()
        combined_boxes = []
        for b in pool:
            try:
                key = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
            except Exception:
                continue
            if key not in seen:
                seen.add(key)
                combined_boxes.append(key)
    except Exception:
        combined_boxes = match_boxes or valid_edge_boxes or []

    # STEP 7: Merge close/overlapping boxes
    try:
        merged_boxes = boxes.merge_close(
            combined_boxes or [], 
            iou_thresh=merge_iou_cfg, 
            gap_thresh=merge_gap_cfg
        )
    except Exception:
        merged_boxes = combined_boxes or []

    # STEP 8: Final filter (minimum match points required)
    try:
        final_boxes = []
        if match_points:
            for box in (merged_boxes or []):
                count = 0
                x0, y0, x1, y1 = box[0], box[1], box[0] + box[2], box[1] + box[3]
                for (mx, my) in match_points:
                    if mx >= x0 and mx < x1 and my >= y0 and my < y1:
                        count += 1
                        if count >= min_pts_cfg:
                            break
                if count >= min_pts_cfg:
                    final_boxes.append(box)
        else:
            final_boxes = []
    except Exception:
        final_boxes = []

    # Save final debug overlays
    _save_pipeline_debug_overlays(label_img, combined_boxes, merged_boxes, 
                                 final_boxes, ref_icon, debug_dir, dbg)

    # Store intermediate results in debug dict
    dbg['match_points'] = match_points
    dbg['combined_boxes'] = combined_boxes
    dbg['merged_combined'] = merged_boxes
    dbg['kept_boxes'] = final_boxes
    
    return final_boxes, dbg


# match_page_candidates and match_point_candidates removed: callers should use
# `match_icon_candidates` for per-icon generation and `feature_matching.match_point_candidates`
# directly when point-only behavior is required. Keeping these wrappers caused
# duplication and API surface area we no longer want to maintain.


def _save_all_edge_boxes_debug(label_img, edge_boxes, ref_icon, debug_dir, dbg):
    """Save debug overlay showing all computed edge boxes."""
    if not debug_dir:
        return
        
    try:
        debug.ensure_dir(debug_dir)
        icon_base = os.path.splitext(os.path.basename(getattr(ref_icon, 'filename', 'icon')))[0]
        
        vis_all_edges = label_img.copy()
        debug.draw_boxes(vis_all_edges, edge_boxes or [], (0, 255, 0), 1)
        
        all_edges_path = os.path.join(debug_dir, f"{icon_base}_all_edge_boxes.png")
        debug.safe_imwrite(all_edges_path, vis_all_edges)
        dbg['all_edge_boxes_overlay'] = all_edges_path
    except Exception:
        pass


def _save_edge_filter_debug(label_img, edge_boxes_filtered, valid_edge_boxes, 
                          match_boxes_raw, ref_icon, debug_dir, dbg, merge_gap_cfg):
    """Save debug overlays for edge filtering steps."""
    if not debug_dir:
        return
        
    try:
        debug.ensure_dir(debug_dir)
        icon_base = os.path.splitext(os.path.basename(getattr(ref_icon, 'filename', 'icon')))[0]
        
        # Filtered edge boxes overlay
        vis_filtered = label_img.copy()
        debug.draw_boxes(vis_filtered, edge_boxes_filtered or [], (0, 255, 0), 2)
        filtered_path = os.path.join(debug_dir, f"{icon_base}_edge_filtered_firstpass.png")
        debug.safe_imwrite(filtered_path, vis_filtered)
        dbg['edge_filtered_firstpass'] = filtered_path
        
        # Nearby clusters overlay
        vis_nearby = label_img.copy()
        for edge_box in (edge_boxes_filtered or []):
            # Draw edge box in green
            debug.draw_boxes(vis_nearby, [edge_box], (0, 255, 0), 2)
            # Draw nearby match clusters in blue
            for match_box in (match_boxes_raw or []):
                if (boxes.iou(edge_box, match_box) > 0.0 or 
                    boxes.rect_gap(edge_box, match_box) <= merge_gap_cfg):
                    debug.draw_boxes(vis_nearby, [match_box], (255, 0, 0), 1)
        
        nearby_path = os.path.join(debug_dir, f"{icon_base}_edge_nearby_clusters.png")
        debug.safe_imwrite(nearby_path, vis_nearby)
        dbg['edge_nearby_clusters'] = nearby_path
        
    except Exception:
        pass


def _save_pipeline_debug_overlays(label_img, combined_boxes, merged_boxes, 
                                final_boxes, ref_icon, debug_dir, dbg):
    """Save debug overlays for pipeline steps."""
    if not debug_dir:
        return
        
    try:
        debug.ensure_dir(debug_dir)
        icon_base = os.path.splitext(os.path.basename(getattr(ref_icon, 'filename', 'icon')))[0]
        
        # Combined overlay
        vis_combined = label_img.copy()
        debug.draw_boxes(vis_combined, combined_boxes or [], (255, 0, 0), 2)
        combined_path = os.path.join(debug_dir, f"{icon_base}_combined.png")
        debug.safe_imwrite(combined_path, vis_combined)
        dbg['combined_overlay'] = combined_path
        
        # Merged overlay
        vis_merged = label_img.copy()
        debug.draw_boxes(vis_merged, merged_boxes or [], (0, 255, 0), 2)
        merged_path = os.path.join(debug_dir, f"{icon_base}_merged.png")
        debug.safe_imwrite(merged_path, vis_merged)
        dbg['merged_overlay'] = merged_path
        
        # Final kept overlay
        vis_kept = label_img.copy()
        debug.draw_boxes(vis_kept, final_boxes or [], (255, 0, 255), 2)
        kept_path = os.path.join(debug_dir, f"{icon_base}_kept.png")
        debug.safe_imwrite(kept_path, vis_kept)
        dbg['kept_overlay'] = kept_path
        
    except Exception:
        pass


def _save_page_debug_overlay(label_img, page_boxes_accum, merged_across_icons, 
                           final_boxes, debug_dir, dbg):
    """Save debug overlay for page-level processing."""
    if not debug_dir:
        return
        
    try:
        debug.ensure_dir(debug_dir)
        seq = debug.next_debug_seq()
        
        vis = label_img.copy()
        # Accumulated boxes in blue
        debug.draw_boxes(vis, page_boxes_accum or [], (255, 0, 0), 1)
        # Merged across icons in yellow
        debug.draw_boxes(vis, merged_across_icons or [], (0, 255, 255), 2)
        # Final boxes in cyan
        debug.draw_boxes(vis, final_boxes or [], (255, 255, 0), 2)
        
        path = os.path.join(debug_dir, f'page_match_candidates_combined_s{seq}.png')
        debug.safe_imwrite(path, vis)
        dbg['page_combined_overlay'] = path
    except Exception:
        pass


# Export public API (removed page/point-level wrappers)
__all__ = ['generate_candidates', 'match_icon_candidates']