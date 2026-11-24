from __future__ import annotations

import numpy as np
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - exercised only in minimal installs
    from itertools import permutations
    def linear_sum_assignment(cost_matrix):
        """Small exact fallback for demos/tests when SciPy is unavailable."""
        c = np.asarray(cost_matrix, float); nr, nc = c.shape
        if nr == 0 or nc == 0: return np.array([], dtype=int), np.array([], dtype=int)
        if max(nr, nc) > 8: raise ImportError("SciPy is required for assignments larger than 8x8")
        if nr <= nc:
            best = min(permutations(range(nc), nr), key=lambda p: sum(c[i, p[i]] for i in range(nr)))
            return np.arange(nr), np.asarray(best, dtype=int)
        cols = min(permutations(range(nr), nc), key=lambda p: sum(c[p[j], j] for j in range(nc)))
        return np.asarray(cols, dtype=int), np.arange(nc)


def cosine_similarity(a, b):
    if a is None or b is None:
        return None
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return None if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1); area_b = max(0, bx2-bx1) * max(0, by2-by1)
    return float(inter / (area_a + area_b - inter)) if area_a + area_b - inter else 0.0


def gallery_similarity(embedding, gallery):
    scores = [cosine_similarity(embedding, g) for g in gallery if embedding is not None and g is not None]
    return max(scores) if scores else None


def associate(detections, tracks, *, appearance_weight=.55, motion_weight=.35, iou_weight=.10, gate=9.21, appearance_threshold=.55, ambiguity_margin=.08):
    if not detections or not tracks:
        return [], list(range(len(tracks))), list(range(len(detections))), np.empty((len(tracks), len(detections)))
    costs = np.full((len(tracks), len(detections)), np.inf)
    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            if track.class_name != det.class_name:
                continue
            maha = track.kalman.mahalanobis_squared(det.world_position if det.world_position is not None else det.center)
            if maha > gate:
                continue
            app = gallery_similarity(det.embedding, track.gallery)
            terms, weights = [], []
            if app is not None:
                terms.append(1 - app); weights.append(appearance_weight)
            terms.append(min(maha / gate, 1.0)); weights.append(motion_weight)
            iou = bbox_iou(track.last_detection.bbox, det.bbox) if track.last_detection is not None else 0.0
            terms.append(1 - iou); weights.append(iou_weight)
            costs[i, j] = float(np.average(terms, weights=weights))
    rows, cols = linear_sum_assignment(np.where(np.isfinite(costs), costs, 1e6))
    matches, used_r, used_c = [], set(), set()
    for r, c in zip(rows, cols):
        if not np.isfinite(costs[r, c]):
            continue
        if detections[c].embedding is not None and tracks[r].gallery:
            s = gallery_similarity(detections[c].embedding, tracks[r].gallery)
            if s is not None and s < appearance_threshold:
                continue
        alternatives = sorted(costs[r, k] for k in range(costs.shape[1]) if np.isfinite(costs[r, k]) and k != c)
        if alternatives and alternatives[0] - costs[r, c] < ambiguity_margin:
            continue
        matches.append((r, c)); used_r.add(r); used_c.add(c)
    return matches, [i for i in range(len(tracks)) if i not in used_r], [j for j in range(len(detections)) if j not in used_c], costs
