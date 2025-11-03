# object_detector.py

import cv2
import time
import numpy as np
import torch # Required for the dfl function in post-processing

# Import the component classes and utilities
from .detector_util import Sort, MetricsLogger, BBoxUtils, KalmanBoxTracker

from .py_utils.coco_utils import COCO_test_helper
from .py_utils.rknn_executor import RKNN_model_container

class ObjectDetector:

    #The main class to handle object detection, tracking, and counting in a video stream.
    #It orchestrates the pre-processing, inference, post-processing, tracking, and visualization.

    def __init__(self, model_path, img_size=(640, 640), obj_thresh=0.048, nms_thresh=0.048,
        max_age=5, min_hits=1, diou_threshold=0.3, dij_threshold=0.9):
        # --- Configuration ---
        self.IMG_SIZE = img_size
        self.OBJ_THRESH = obj_thresh
        self.NMS_THRESH = nms_thresh
        np.random.seed(0)

        # --- Initialization of Components ---

        print("INFO: Initializing model...")
        self.model = RKNN_model_container(model_path, 'rk3588')
        self.mot_tracker = Sort(max_age=max_age, min_hits=min_hits, diou_threshold=diou_threshold, dij_threshold=dij_threshold) # Using more robust parameters
        self.metrics_logger = MetricsLogger('detection_metrics.csv')
        self.co_helper = COCO_test_helper(enable_letter_box=True)
        print("INFO: Model and components initialized.")

        # --- State Variables for Tracking and Counting ---
        self.frame_counter = 0
        self.previous_track_2 = np.empty((0,5))
        self.previous_track_3 = np.empty((0,5))

        # --- State Variables for Multiple Counting Methods ---
        self.fish_count_1 = 0
        self.counted_ids_1 = set()
        self.fish_count_2 = 0
        self.counted_ids_2 = set()
        self.fish_count_3 = 0
        self.counted_ids_3 = set()

        # --- Define Counting Boundaries ---
        # PERBAIKAN: Gunakan IMG_SIZE[1] (tinggi) dan ganti nama variabel
        self.line_y_pos = int(0.43 * self.IMG_SIZE[1])
        self.line_y_pos_bottom = int(0.4 * self.IMG_SIZE[1])
        self.line_y_pos_top = int(0.3 * self.IMG_SIZE[1])


    # ====================================================================
    # fOR RESETING THE DETECTOR STATE
    # ====================================================================
    def reset(self):
        #Full reset to match original state"""
        self.fish_count_1 = 0
        self.fish_count_2 = 0
        self.fish_count_3 = 0
        self.counted_ids_1.clear()
        self.counted_ids_2.clear()
        self.counted_ids_3.clear()
        self.previous_track_2 = np.empty((0,5))
        self.previous_track_3 = np.empty((0,5))
        self.frame_counter = 0
    # ====================================================================

    def process_frame(self, frame):

        #Accepts a single frame, performs detection and tracking, and returns the annotated frame.
        #:param frame: A single video frame (NumPy array).
        #:return: A tuple containing (annotated_frame, dictionary_of_counts).

        self.frame_counter += 1

        # --- 1. Pre-processing ---
        start_time = time.time()
        img_processed, original_frame = self._preprocess(frame)
        preprocess_time = time.time() - start_time

        # --- 2. Model Inference ---
        start_time = time.time()
        model_outputs = self.model.run([img_processed])
        detection_time = time.time() - start_time

        # --- 3. Post-processing ---
        start_time = time.time()
        boxes, _, scores = self._post_process(model_outputs)
        postprocess_time = time.time() - start_time

        # --- 4. Tracking ---
        start_time = time.time()
        detections = np.column_stack((boxes, scores)) if boxes.size > 0 else np.empty((0, 5))
        active_tracks = self.mot_tracker.update(detections)
        tracking_time = time.time() - start_time

        # --- 5. Counting and Visualization ---
        start_time = time.time()
        # Use the resized original frame for visualization to match box coordinates
        annotated_frame = self._update_counts_and_visualize(original_frame, active_tracks, boxes)
        counting_time = time.time() - start_time

        # --- 6. Logging (Optional) ---
        # self.metrics_logger.log(...)

        counts = {
            "count_1": self.fish_count_1,
            "count_2": self.fish_count_2,
            "count_3": self.fish_count_3
        }
        return annotated_frame, counts

    def _preprocess(self, frame):
        #Prepares a frame for the model.
        # The letter_box function returns the padded image and the original resized image
        img_padded = self.co_helper.letter_box(im=frame.copy(), new_shape=self.IMG_SIZE, pad_color=(0,0,0))
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        # We also need a version of the original frame resized to the display size for drawing


        return img_rgb, frame.copy()

    def _update_counts_and_visualize(self, frame, tracks, detected_boxes):
        #Updates all fish counts and draws visualizations on the frame.
        vis_frame = frame.copy()
        # Draw all boundary lines
        # Draw all boundary lines
        frame_width = vis_frame.shape[1]
        # PERBAIKAN: Gambar garis horizontal dengan benar
        cv2.line(vis_frame, (0, self.line_y_pos), (frame_width, self.line_y_pos), (0, 255, 0), 3)



        # Draw raw detections in green
        for box in detected_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Draw active tracks in blue
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)

            # Draw bounding box and ID
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(vis_frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # --- Counting Method 1: Simple Right Boundary Cross ---
            if track_id not in self.counted_ids_1 and y2 > self.line_y_pos: #rotasi ubah jadi y
                self.fish_count_1 += 1
                self.counted_ids_1.add(track_id)

            # --- Counting Method 2: Enter Region ---
            if track_id not in self.counted_ids_2 and y2 > self.line_y_pos_top and y1 < self.line_y_pos_bottom: #rotasi ubah jadi y
                self.fish_count_2 += 1
                self.counted_ids_2.add(track_id)

             # --- Counting Method 3: Robust Boundary Crossing (Bottom-to-Up Movement) ---
            if track_id not in self.counted_ids_3:
                # Cari posisi track ini di frame sebelumnya
                prev_track_match = self.previous_track_2[self.previous_track_2[:, 4] == track_id]

                if len(prev_track_match) > 0:
                    # Ambil koordinat y dari tepi ATAS box sebelumnya
                    prev_y1 = int(prev_track_match[0][1])

                    # Ambil koordinat y dari tepi ATAS box saat ini (y1 sudah ada dari `map(int, track)`)
                    current_y1 = int(y1)

                    # Ambil posisi garis hitung
                    counting_line_y = self.line_y_pos

                    # KONDISI DIUBAH: Cek jika box melintasi garis dari BAWAH ke ATAS
                    # Logikanya: Posisi atas box sebelumnya harus di bawah garis (nilai Y lebih besar)
                    # dan posisi atas box sekarang harus di atas garis (nilai Y lebih kecil).
                    if prev_y1 > counting_line_y and current_y1 <= counting_line_y:
                        self.fish_count_3 += 1
                        self.counted_ids_3.add(track_id)
                        # Opsional: Beri warna khusus pada box yang baru terhitung
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 255), 2) # Warna kuning


        # --- FIX: Update historical track data using .copy() to prevent reference errors ---
        self.previous_track_3 = self.previous_track_2
        self.previous_track_2 = tracks.copy() if len(tracks) > 0 else np.empty((0, 5))

        # Draw counters on frame
        cv2.putText(vis_frame, f'Frame: {self.frame_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_frame, f'Method 1: {self.fish_count_1}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_frame, f'Method 2: {self.fish_count_2}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_frame, f'Method 3: {self.fish_count_3}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return vis_frame

    def _check_cross(self, boundary, start_centroid, current_centroid):
        #Helper function to check if a line segment intersects a boundary line.
        x0_0, y0_0 = boundary[0]
        x1_0, y1_0 = boundary[1]
        x0_1, y0_1 = start_centroid
        x1_1, y1_1 = current_centroid

        dx0 = x1_0 - x0_0
        dy0 = y1_0 - y0_0
        dx1 = x1_1 - x0_1
        dy1 = y1_1 - y0_1

        denominator = dx1 * dy0 - dy1 * dx0
        if denominator == 0:
            return False

        t = ((x0_0 - x0_1) * dy1 - (y0_0 - y0_1) * dx1) / denominator
        u = ((x0_0 - x0_1) * dy0 - (y0_0 - y0_1) * dx0) / denominator
        if 0 <= t <= 1 and 0 <= u <= 1:
            return y1_1 > y0_1  # Only count if moving up #rotasi ubah jadi y
        return False

    # --- Post-processing methods specific to the YOLOv6 model ---
    def _post_process(self, input_data):
        #Converts raw model output into filtered bounding boxes, classes, and scores.
        boxes, scores, classes_conf = [], [], []
        default_branch=3
        pair_per_branch = len(input_data)//default_branch
        for i in range(default_branch):
            boxes.append(self._box_process(input_data[pair_per_branch*i]))
            classes_conf.append(input_data[pair_per_branch*i+1])
            scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0,2,3,1)
            return _in.reshape(-1, ch)

        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]

        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)

        boxes, classes, scores = self._filter_boxes(boxes, scores, classes_conf)

        nboxes, nclasses, nscores = [], [], []
        # The model only detects one class, so we don't need to loop
        if boxes.size > 0:
            keep = self._nms_boxes(boxes, scores)
            if len(keep) != 0:
                nboxes.append(boxes[keep])
                nclasses.append(classes[keep])
                nscores.append(scores[keep])

        if not nclasses and not nscores:
            return np.array([]), np.array([]), np.array([])

        return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)

    def _filter_boxes(self, boxes, box_confidences, box_class_probs):
        box_confidences = box_confidences.reshape(-1)
        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)
        _class_pos = np.where(class_max_score * box_confidences >= self.OBJ_THRESH)
        return boxes[_class_pos], classes[_class_pos], (class_max_score * box_confidences)[_class_pos]

    def _nms_boxes(self, boxes, scores):
        x, y = boxes[:, 0], boxes[:, 1]
        w, h = boxes[:, 2] - x, boxes[:, 3] - y
        areas = w * h
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
            w1 = np.maximum(0.0, xx2 - xx1)
            h1 = np.maximum(0.0, yy2 - yy1)
            inter = w1 * h1
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
            inds = np.where(ovr <= self.NMS_THRESH)[0]
            order = order[inds + 1]
        return np.array(keep)

    def _box_process(self, position):
       #Original box_process implementation
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([self.IMG_SIZE[1]//grid_h, self.IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

        if position.shape[1] == 4:
            box_xy  = grid +0.5 -position[:,0:2,:,:]
            box_xy2 = grid +0.5 +position[:,2:4,:,:]
            xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)
        else:
            position = self._dfl(position)
            box_xy  = grid +0.5 -position[:,0:2,:,:]
            box_xy2 = grid +0.5 +position[:,2:4,:,:]
            xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

        return xyxy
    @staticmethod
    def _dfl(position):
        # Distribution Focal Loss
        x = torch.tensor(position)
        n,c,h,w = x.shape
        p_num = 4
        mc = c//p_num
        y = x.reshape(n,p_num,mc,h,w).softmax(2)
        acc_metrix = torch.arange(mc).float().reshape(1,1,mc,1,1)
        return (y*acc_metrix).sum(2).numpy()
