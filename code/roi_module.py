'''Region Of Interest module'''

import dlib
import cv2
from collections import OrderedDict
from imutils import face_utils
import numpy as np


FACIAL_LANDMARKS_81_IDXS = OrderedDict([("mouth", (48, 68)),
                                        ("right_eyebrow", (17, 22)),
                                        ("left_eyebrow", (22, 27)),
                                        ("right_eye", (36, 42)),
                                        ("left_eye", (42, 48)),
                                        ("nose", (27, 36)),
                                        ("jaw", (0, 17)),
                                        ("forehead", (68, 81))])


COLORS = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
          (168, 100, 168), (158, 163, 32), (163, 38, 32),
          (180, 42, 220), (111, 19, 33)]


DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor('shape_predictor_81_face_landmarks.dat')


def rect_to_bb(rect):
    """Compute bounding box parameters of rect"""
    x_coord = rect.left()
    y_coord = rect.top()
    width = rect.right() - x_coord
    height = rect.bottom() - y_coord
    return (x_coord, y_coord, width, height)


def rectangles(i, RECT, FRAME, bpm):
    (x, y, w, h) = rect_to_bb(RECT)
    cv2.rectangle(FRAME, (x, y), (x+w, y+h), (0, 255, 255), 2)
    cv2.putText(FRAME, "FACE #{} BPM: {}".format(i + 1, bpm), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def crop_face(RECT, FRAME):
    (x, y, w, h) = rect_to_bb(RECT)
    return FRAME[y:y+h, x:x+w]


def feature_detection_forehead(PREDICTOR, FRAME_ORIG, FRAME, RECT):
    SHAPE = PREDICTOR(FRAME_ORIG, RECT)
    SHAPE = face_utils.shape_to_np(SHAPE)

    left = SHAPE[21][0]
    right = SHAPE[22][0]
    top = min(SHAPE[21][1], SHAPE[22][1]) - (right - left)
    bottom = max(SHAPE[21][1], SHAPE[22][1])
    cv2.rectangle(FRAME, (left, bottom-10), (right, top-10), (0, 0, 255), 2)

    return FRAME_ORIG[top:bottom, left:right]

def feature_detection_face(PREDICTOR, FRAME_ORIG, FRAME, RECT):
    SHAPE = PREDICTOR(FRAME_ORIG, RECT)
    SHAPE = face_utils.shape_to_np(SHAPE)

    left = SHAPE[36][0]
    right = SHAPE[45][0]
    top = SHAPE[28][1]
    bottom = SHAPE[30][1]
    cv2.rectangle(FRAME, (left, bottom-15), (right, top-15), (0, 0, 255), 2)

    return FRAME_ORIG[top:bottom, left:right]
