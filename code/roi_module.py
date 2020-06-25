'''Region Of Interest module'''

import dlib
import cv2
from collections import OrderedDict
from imutils import face_utils
import numpy as np


def rect_to_bb(rect):
    """Compute bounding box parameters of rect"""
    x_coord = rect.left()
    y_coord = rect.top()
    width = rect.right() - x_coord
    height = rect.bottom() - y_coord
    return (x_coord, y_coord, width, height)


def rectangles(i, RECT, FRAME):
    """Draw rectangle around face coordinates"""
    (x, y, w, h) = rect_to_bb(RECT)
    cv2.rectangle(FRAME, (x, y), (x+w, y+h), (0, 255, 255), 2)
    cv2.putText(FRAME, "FACE #{}".format(i + 1), (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
                False)


def roi_extraction(PREDICTOR, FRAME_ORIG, FRAME, RECT):
    SHAPE = PREDICTOR(FRAME_ORIG, RECT)
    SHAPE = face_utils.shape_to_np(SHAPE)

    left = SHAPE[21][0]
    right = SHAPE[22][0]
    top = min(SHAPE[21][1], SHAPE[22][1]) - (right - left)
    bottom = max(SHAPE[21][1], SHAPE[22][1])

    cv2.rectangle(FRAME, (left, bottom-10), (right, top-10), (0, 0, 255), 2)
    MEAN = np.mean(FRAME_ORIG[top:bottom, left:right, 1])

    left = SHAPE[36][0]
    right = SHAPE[45][0]
    top = SHAPE[28][1]
    bottom = SHAPE[30][1]

    cv2.rectangle(FRAME, (left, bottom-15), (right, top-15), (0, 0, 255), 2)
    MEAN += np.mean(np.array(FRAME_ORIG[top:bottom, left:right, 1]))

    return MEAN / 2
