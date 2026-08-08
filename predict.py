"""Live traffic sign detection and classification from a webcam or video file.

Pipeline per frame:
  1. detect.find_candidate_regions() proposes candidate boxes using a classic
     color/shape heuristic (not a trained detector - see detect.py).
  2. Each candidate crop is resized to 50x50 and classified with the trained
     CNN loaded from --model.
  3. Boxes/labels are drawn on the frame when confidence exceeds --threshold.

Usage:
  python predict.py --model models/complex_model.keras --source 0 --threshold 0.6
Press 'q' to quit.
"""

import argparse

import cv2
import numpy as np

from detect import find_candidate_regions
from utils import classes

INPUT_SIZE = (50, 50)


def parse_args():
    parser = argparse.ArgumentParser(description="Live traffic sign detection and classification")
    parser.add_argument("--model", default="models/complex_model.keras",
                         help="Path to a trained Keras model (default: models/complex_model.keras)")
    parser.add_argument("--source", default="0",
                         help="Video source: webcam index (e.g. 0) or path to a video file (default: 0)")
    parser.add_argument("--threshold", type=float, default=0.6,
                         help="Minimum classification confidence to draw a label (default: 0.6)")
    return parser.parse_args()


def resolve_source(source):
    """Webcam indices arrive as strings ('0'); video paths stay as strings."""
    try:
        return int(source)
    except ValueError:
        return source


def classify_crop(model, frame, box):
    x, y, w, h = box
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        return None, 0.0

    resized = cv2.resize(crop, INPUT_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    batch = np.expand_dims(normalized, axis=0)

    predictions = model.predict(batch, verbose=0)[0]
    class_id = int(np.argmax(predictions))
    confidence = float(predictions[class_id])
    return class_id, confidence


def annotate_frame(frame, box, class_id, confidence):
    x, y, w, h = box
    label = "{0} ({1:.0%})".format(classes.get(class_id, "Unknown"), confidence)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, label, (x, max(y - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)


def main():
    args = parse_args()

    from tensorflow.keras.models import load_model
    model = load_model(args.model)

    capture = cv2.VideoCapture(resolve_source(args.source))
    if not capture.isOpened():
        raise RuntimeError("Could not open video source: {0}".format(args.source))

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            for box in find_candidate_regions(frame):
                class_id, confidence = classify_crop(model, frame, box)
                if class_id is not None and confidence >= args.threshold:
                    annotate_frame(frame, box, class_id, confidence)

            cv2.imshow("Traffic Sign Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
