"""CLI: detect faces with YOLOv4 and predict age for each one."""

import argparse

import cv2
import torch

from face_detector import FaceDetector
from age_estimator import AgeEstimator


def parse_args():
    p = argparse.ArgumentParser(description="Face detection + age prediction")
    p.add_argument("--image", required=True, help="path to input image")
    p.add_argument("--yolo-cfg", required=True)
    p.add_argument("--yolo-weights", required=True)
    p.add_argument("--age-model", required=True)
    p.add_argument("--output", default="out.jpg")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--cuda", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    detector = FaceDetector(args.yolo_cfg, args.yolo_weights, use_cuda=args.cuda)
    age_model = AgeEstimator.load(args.age_model, device=device)

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)

    boxes = detector.detect(image, conf_threshold=args.conf)
    print(f"Detected {len(boxes)} face(s)")

    for (x, y, w, h) in boxes:
        x, y = max(0, x), max(0, y)
        crop = image[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        age = age_model.predict(crop, device=device)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"Age: {age}", (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        print(f"  box=({x},{y},{w},{h}) age={age}")

    cv2.imwrite(args.output, image)
    print(f"Saved annotated image to {args.output}")


if __name__ == "__main__":
    main()
