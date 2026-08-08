"""Heuristic candidate-region detector for traffic signs.

NOTE: This is a classic color/shape heuristic (HSV color masking + contour
filtering), not a trained detector. It is a reasonable, cheap front-end to
feed candidate crops into a CNN classifier, but it will miss signs whose
color falls outside the red/blue masks and will produce false positives on
other red/blue objects. A trained detector model (e.g. a small object
detection network) would be a good future upgrade.
"""

import cv2
import numpy as np

# HSV ranges for typical European traffic sign colors.
# Red wraps around the hue circle, so it needs two ranges.
_RED_RANGES = [
    ((0, 70, 50), (10, 255, 255)),
    ((170, 70, 50), (180, 255, 255)),
]
_BLUE_RANGE = ((100, 70, 50), (130, 255, 255))

_MIN_AREA = 400
_MIN_ASPECT = 0.6
_MAX_ASPECT = 1.7


def _color_mask(hsv):
    mask = cv2.inRange(hsv, np.array(_BLUE_RANGE[0]), np.array(_BLUE_RANGE[1]))
    for lower, upper in _RED_RANGES:
        mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    return mask


def find_candidate_regions(frame, min_area=_MIN_AREA):
    """Find candidate traffic-sign regions in a BGR frame.

    Uses HSV color thresholding for typical sign colors (red/blue), then
    filters contours by area and aspect ratio to approximate the roughly
    circular/triangular/square shape of a sign. This is a heuristic, not a
    learned detector.

    Parameters:
    frame (np.ndarray): BGR image (as read by cv2.VideoCapture)
    min_area (int): minimum contour bounding-box area in pixels to keep

    Return:
    list[tuple[int, int, int, int]]: candidate boxes as (x, y, w, h)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = _color_mask(hsv)

    # close small gaps and remove speckle noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < min_area or h == 0:
            continue
        aspect = w / float(h)
        if aspect < _MIN_ASPECT or aspect > _MAX_ASPECT:
            continue
        boxes.append((x, y, w, h))

    return boxes
