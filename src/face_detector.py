"""YOLOv4-based face detector using OpenCV's DNN module."""

import cv2
import numpy as np


class FaceDetector:
    def __init__(self, cfg_path, weights_path, input_size=416, use_cuda=False):
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        if use_cuda:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.input_size = input_size
        self.layer_names = self.net.getUnconnectedOutLayersNames()

    def detect(self, image, conf_threshold=0.5, nms_threshold=0.4):
        """Returns a list of (x, y, w, h) boxes for detected faces."""
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image, 1 / 255.0, (self.input_size, self.input_size),
            swapRB=True, crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.layer_names)

        boxes, confidences = [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                confidence = scores[0] if len(scores) else detection[4]
                if confidence > conf_threshold:
                    cx, cy, w, h = detection[0:4] * np.array(
                        [width, height, width, height]
                    )
                    x = int(cx - w / 2)
                    y = int(cy - h / 2)
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        if len(indices) == 0:
            return []
        indices = np.array(indices).flatten()
        return [tuple(boxes[i]) for i in indices]
