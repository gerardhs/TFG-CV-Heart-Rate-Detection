""" Real-time facial landmark detection using DLIB, OpenCV and imutils """

from collections import OrderedDict
import argparse
from imutils.video import VideoStream
import imutils
from imutils import face_utils
import dlib
import cv2


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


def rect_to_bb(rect):
    """Compute bounding box parameters of rect"""
    x_coord = rect.left()
    y_coord = rect.top()
    width = rect.right() - x_coord
    height = rect.bottom() - y_coord
    return (x_coord, y_coord, width, height)


AP = argparse.ArgumentParser()
AP.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")

ARGS = vars(AP.parse_args())
print("[INFO] Loading facial landmark predictor...")
DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor(ARGS["shape_predictor"])
print("[INFO] Done")

VS = VideoStream().start()

while True:
    FRAME = VS.read()
    FRAME = imutils.resize(FRAME, width=720)
    GRAY = cv2.cvtColor(FRAME, cv2.COLOR_BGR2GRAY)
    RECTS = DETECTOR(GRAY, 0)

    for (i, RECT) in enumerate(RECTS):

        (x, y, w, h) = rect_to_bb(RECT)
        cv2.rectangle(FRAME, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(FRAME, "FACE #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        SHAPE = PREDICTOR(GRAY, RECT)
        SHAPE = face_utils.shape_to_np(SHAPE)
        overlay = FRAME.copy()

        for (x, y) in SHAPE:
            cv2.circle(FRAME, (x, y), 1, (0, 0, 255), -1)

        for (l, name) in enumerate(FACIAL_LANDMARKS_81_IDXS.keys()):
            (j, k) = FACIAL_LANDMARKS_81_IDXS[name]
            pts = SHAPE[j:k]
            if name == "jaw":
                for p in range(1, len(pts)):
                    pt_A = tuple(pts[p-1])
                    pt_B = tuple(pts[p])
                    cv2.line(FRAME, pt_A, pt_B, COLORS[l], 2)

            elif name != "forehead":
                hull = cv2.convexHull(pts)
                cv2.drawContours(FRAME, [hull], -1, COLORS[l], -1)

        RIGHT_CHEEK = \
            overlay[SHAPE[29][1]:SHAPE[33][1], SHAPE[54][0]:SHAPE[12][0]]

        LEFT_CHEEK = \
            overlay[SHAPE[29][1]:SHAPE[33][1], SHAPE[4][0]:SHAPE[48][0]]

        cv2.rectangle(FRAME, (SHAPE[54][0], SHAPE[29][1]),
                      (SHAPE[12][0], SHAPE[33][1]), (0, 0, 255), 2)

        cv2.rectangle(FRAME, (SHAPE[4][0], SHAPE[29][1]),
                      (SHAPE[48][0], SHAPE[33][1]), (0, 0, 255), 2)

        cv2.rectangle(FRAME, (SHAPE[19][0], SHAPE[19][1]),
                      (SHAPE[24][0], SHAPE[75][1]), (0, 0, 255), 2)

    cv2.imshow("Frame", FRAME)
    KEY = cv2.waitKey(1) & 0xFF

    if KEY == ord("q"):
        break

cv2.destroyAllWindows()
VS.stop()
