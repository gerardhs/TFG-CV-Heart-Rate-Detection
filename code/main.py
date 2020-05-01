''' Live capture using webcam '''


import cv2
import dlib
import roi_module
import imutils
import numpy as np
from evm_module import Eulerian_Video_Magnification

CAP = cv2.VideoCapture(0)

if not CAP.isOpened():
    raise IOError("Cannot open webcam")

DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor('shape_predictor_81_face_landmarks.dat')
ALIGN_PREDICTOR = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = imutils.face_utils.FaceAligner(ALIGN_PREDICTOR, desiredFaceWidth=256)
EVM = Eulerian_Video_Magnification(lvl=3, amplification=20,
                                   fps=4, frame_buffer_size=10)


while True:
    CAP.set(cv2.CAP_PROP_FPS,30)
    fps = CAP.get(cv2.CAP_PROP_FPS)
    RET, FRAME = CAP.read()
    FRAME_ORIG = cv2.resize(FRAME, None, fx=1, fy=1,
                            interpolation=cv2.INTER_AREA)
    EVM.frames.append(FRAME)
    #GAUSSIAN_FRAME = \
    #    evm_module.gaussian_pyramid(FRAME_ORIG, 3)

    FRAME = FRAME_ORIG.copy()
    GRAY = cv2.cvtColor(FRAME, cv2.COLOR_BGR2GRAY)
    RECTS = DETECTOR(GRAY, 0)
    for (i, RECT) in enumerate(RECTS):
        roi_module.rectangles(i, RECT, FRAME)
        FOREHEAD = roi_module.feature_detection(PREDICTOR,
                                                FRAME_ORIG, FRAME, RECT)
        cv2.imshow("Forehead #{}".format(i+1), FOREHEAD)
    cv2.imshow('Live face detection', FRAME)
    EVM.build_pyramids(FRAME_ORIG)
    cv2.imshow('EVM test', EVM.reconnstruct(EVM.pyramids[-1]))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

CAP.release()
cv2.destroyAllWindows()
