'''Region Of Interest module'''

import dlib
import cv2
from collections import OrderedDict
from imutils import face_utils


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


def rectangles(i, RECT, FRAME):
    (x, y, w, h) = rect_to_bb(RECT)
    cv2.rectangle(FRAME, (x, y), (x+w, y+h), (0, 255, 255), 2)
    cv2.putText(FRAME, "FACE #{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def feature_detection(PREDICTOR, FRAME_ORIG, FRAME, RECT):
    GRAY = cv2.cvtColor(FRAME_ORIG, cv2.COLOR_BGR2GRAY)
    SHAPE = PREDICTOR(GRAY, RECT)
    SHAPE = face_utils.shape_to_np(SHAPE)

    FOREHEAD = \
        FRAME_ORIG[SHAPE[75][1]:SHAPE[19][1], SHAPE[21][0]:SHAPE[22][0]]

    cv2.rectangle(FRAME, (SHAPE[54][0], SHAPE[29][1]),
                  (SHAPE[12][0], SHAPE[33][1]), (0, 0, 255), 2)

    cv2.rectangle(FRAME, (SHAPE[4][0], SHAPE[29][1]),
                  (SHAPE[48][0], SHAPE[33][1]), (0, 0, 255), 2)

    cv2.rectangle(FRAME, (SHAPE[21][0], SHAPE[19][1]),
                  (SHAPE[22][0], SHAPE[75][1]), (0, 0, 255), 2)

    return FOREHEAD
