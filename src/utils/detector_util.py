# detector_util.py

import numpy as np
import math
import csv
import time
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# ============================================================================
# 1. UTILITY CLASSES
# ============================================================================

class BBoxUtils:
    """
    A utility class for bounding box operations. All methods are static.
    """
    @staticmethod
    def iou_batch(bb_test, bb_gt):
        """Computes IOU between two sets of bounding boxes."""
        if bb_test.size == 0 or bb_gt.size == 0:
            return np.empty((bb_test.shape[0], bb_gt.shape[0]))
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)
        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
                  + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh + 1e-7)
        return o

    @staticmethod
    def convert_bbox_to_xywh(bbox):
        """Converts [x1,y1,x2,y2,id] to [center_x, center_y, w, h, id]."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        return np.array([x, y, w, h, bbox[4]]).reshape((1, 5))

    @staticmethod
    def convert_bbox_to_z(bbox):
        """Converts [x1,y1,x2,y2] to Kalman filter format [center_x, center_y, area, aspect_ratio]."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h
        r = w / float(h + 1e-6)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(x):
        """Converts Kalman filter state [center_x, center_y, area, aspect_ratio] to [x1,y1,x2,y2]."""
        w = np.sqrt(abs(x[2] * x[3])) # Use abs to prevent sqrt of negative
        h = x[2] / (w + 1e-6)
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))

    @staticmethod
    def L_diagonal(x1min, x2min, x1max, x2max, y1min, y2min, y1max, y2max):
        """Calculates the diagonal of the smallest enclosing box of two boxes."""
        xmin = min(x1min, x2min)
        xmax = max(x1max, x2max)
        ymin = min(y1min, y2min)
        ymax = max(y1max, y2max)
        return np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)

    @staticmethod
    def dij_distance(dets, trks):
        """Calculates the normalized central point distance between detections and trackers."""
        if dets.shape[0] == 0 or trks.shape[0] == 0:
            return np.empty((dets.shape[0], trks.shape[0]))

        xdetmin, ydetmin, xdetmax, ydetmax = dets[:,0], dets[:,1], dets[:,2], dets[:,3]
        xtrkmin, ytrkmin, xtrkmax, ytrkmax = trks[:,0], trks[:,1], trks[:,2], trks[:,3]

        x1, y1 = (dets[:,0] + dets[:,2]) / 2, (dets[:,1] + dets[:,3]) / 2
        x2, y2 = (trks[:,0] + trks[:,2]) / 2, (trks[:,1] + trks[:,3]) / 2

        dij_matrix = np.zeros((len(dets), len(trks)))
        for i in range(len(dets)):
            for j in range(len(trks)):
                L = BBoxUtils.L_diagonal(xdetmin[i], xtrkmin[j], xdetmax[i], xtrkmax[j], ydetmin[i], ytrkmin[j], ydetmax[i], ytrkmax[j])
                dist_sq = (x1[i] - x2[j])**2 + (y1[i] - y2[j])**2
                dij_matrix[i, j] = 1 - (dist_sq / (L**2 + 1e-7))
        return dij_matrix

    @staticmethod
    def DIOU_2(iou_matrix, dij_matrix):
        """Calculates the average of IoU and DIJ for a combined metric."""
        return (iou_matrix + dij_matrix) / 2

# ============================================================================
# 2. COMPONENT CLASSES
# ============================================================================

class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects.
    """
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0], [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = BBoxUtils.convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(BBoxUtils.convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(BBoxUtils.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return BBoxUtils.convert_x_to_bbox(self.kf.x)

class Sort:
    """
    A robust implementation of Simple Online and Realtime Tracking (SORT)
    with a three-stage matching cascade.
    """
    def __init__(self, max_age=20, min_hits=3, diou_threshold=0.3, dij_threshold=0.9):
        self.max_age = max_age
        self.min_hits = min_hits
        self.diou_threshold = diou_threshold
        self.dij_threshold = dij_threshold
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0 # Reset tracker count on new instance

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1

        # 1. Predict next state for existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in sorted(to_del, reverse=True):
            self.trackers.pop(t)

        # 2. Divide detections by score
        highscore_dets = dets[dets[:, -1] >= 0.7] if dets.size > 0 else np.empty((0,5))
        lowscore_dets = dets[dets[:, -1] < 0.7] if dets.size > 0 else np.empty((0,5))

        # --- Association Cascade ---
        # 3. First, associate high-score detections with trackers using DIJ distance
        dij_matrix = BBoxUtils.dij_distance(highscore_dets, trks)
        matched1, um_dets1, um_trks1 = self._associate(highscore_dets, trks, dij_matrix, self.dij_threshold)

        # 4. Then, associate low-score detections with remaining unmatched trackers
        um_trks1_arr = trks[um_trks1]
        iou_matrix = BBoxUtils.iou_batch(lowscore_dets, um_trks1_arr)
        dij_matrix_low = BBoxUtils.dij_distance(lowscore_dets, um_trks1_arr)
        diou_matrix = BBoxUtils.DIOU_2(iou_matrix, dij_matrix_low)
        matched2, _, um_trks2_indices = self._associate(lowscore_dets, um_trks1_arr, diou_matrix, self.diou_threshold)
        um_trks2 = um_trks1[um_trks2_indices] # Get original tracker indices

        # 5. Finally, associate remaining high-score detections with remaining trackers from step 4
        um_dets1_arr = highscore_dets[um_dets1]
        um_trks2_arr = trks[um_trks2]
        iou_matrix_rem = BBoxUtils.iou_batch(um_dets1_arr, um_trks2_arr)
        dij_matrix_rem = BBoxUtils.dij_distance(um_dets1_arr, um_trks2_arr)
        diou_matrix_rem = BBoxUtils.DIOU_2(iou_matrix_rem, dij_matrix_rem)
        matched3, um_dets3_indices, _ = self._associate(um_dets1_arr, um_trks2_arr, diou_matrix_rem, self.diou_threshold)

        # 6. Update trackers with their associated detections
        for det_idx, trk_idx in matched1:
            self.trackers[trk_idx].update(highscore_dets[det_idx])
        for det_idx, trk_idx in matched2:
            original_trk_idx = um_trks1[trk_idx]
            self.trackers[original_trk_idx].update(lowscore_dets[det_idx])
        for det_idx, trk_idx in matched3:
            original_det_idx = um_dets1[det_idx]
            original_trk_idx = um_trks2[trk_idx]
            self.trackers[original_trk_idx].update(highscore_dets[original_det_idx])

        # 7. Create new trackers for remaining unmatched high-score detections
        final_um_dets_indices = um_dets1[um_dets3_indices]
        for i in final_um_dets_indices:
            trk = KalmanBoxTracker(highscore_dets[i])
            self.trackers.append(trk)

        # 8. Output active tracks and remove old ones
        ret = []
        for i in range(len(self.trackers) - 1, -1, -1):
            trk = self.trackers[i]
            # Only output tracks that are confirmed (min_hits) and recently updated
            if trk.time_since_update < 1 and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                d = trk.get_state()[0]
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))

            # Remove dead tracklets
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return np.concatenate(ret) if ret else np.empty((0, 5))

    def _associate(self, dets, trks, metric_matrix, threshold):
        """Helper for the Hungarian algorithm-based association."""
        if metric_matrix.size == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(dets), dtype=int), np.arange(len(trks), dtype=int)

        # Use negative matrix because linear_sum_assignment finds minimum cost
        row_ind, col_ind = linear_sum_assignment(-metric_matrix)

        matched_indices = []
        for r, c in zip(row_ind, col_ind):
            if metric_matrix[r, c] >= threshold:
                matched_indices.append([r, c])

        matched_indices = np.array(matched_indices) if matched_indices else np.empty((0, 2))

        unmatched_detections = np.delete(np.arange(len(dets)), matched_indices[:, 0] if matched_indices.size > 0 else [])
        unmatched_trackers = np.delete(np.arange(len(trks)), matched_indices[:, 1] if matched_indices.size > 0 else [])

        return matched_indices.astype(int), unmatched_detections.astype(int), unmatched_trackers.astype(int)


class MetricsLogger:
    """Logs performance metrics to a CSV file."""
    def __init__(self, output_path):
        self.output_path = output_path
        self.file_handle = open(self.output_path, 'w', newline='')
        self.writer = csv.writer(self.file_handle)
        self.writer.writerow(['frame_number', 'preprocess_time', 'detection_time', 'postprocess_time', 'tracking_time', 'counting_time', 'fish_count'])

    def log(self, **kwargs):
        self.writer.writerow(list(kwargs.values()))

    def close(self):
        self.file_handle.close()
