'''Validation with video data'''


import cv2
import dlib
import roi_module
from evm_module import Eulerian_Video_Magnification
import visualization_module
import numpy as np
import time
import matplotlib.pyplot as plt
import heartpy as hp


PREDICTOR = dlib.shape_predictor('shape_predictor_81_face_landmarks.dat')
DETECTOR = dlib.get_frontal_face_detector()


CAP = cv2.VideoCapture('./data/P1LC1/P1LC1_original.MTS')
fps = CAP.get(cv2.CAP_PROP_FPS)

EVM = Eulerian_Video_Magnification(lvl=6, amplification=80,
                                   frame_buffer_size=50, attenuation=1, fps=fps)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []
TIME_START = time.perf_counter()
bpm = 0
iter = 0
while CAP.isOpened():

    RET, FRAME = CAP.read()
    iter = iter + 1
    if RET == 0:
        break
    key = cv2.waitKey(2)
    FRAME_ORIG = cv2.resize(FRAME, None, fx=1, fy=1,
                            interpolation=cv2.INTER_AREA)
    FRAME_ORIG = cv2.resize(FRAME_ORIG[:, :, :], (640, 384))
    FRAME = FRAME_ORIG.copy()
    GRAY = cv2.cvtColor(FRAME, cv2.COLOR_BGR2GRAY)
    RECTS = DETECTOR(GRAY, 0)
    if RECTS:
        EVM.frames.append(FRAME_ORIG)
        EVM.build_pyramids(FRAME_ORIG)
        if len(EVM.frames) > EVM.frame_buffer_size*3:
            FRAME_MAGNIFIED = cv2.convertScaleAbs(EVM.magnify(FRAME_ORIG, 0.7, 3.1))
            cv2.imshow('EVM', FRAME_MAGNIFIED)
            for (i, RECT) in enumerate(RECTS):
                roi_module.rectangles(i, RECT, FRAME, bpm)
                FOREHEAD = roi_module.feature_detection(PREDICTOR,
                                                        FRAME_MAGNIFIED, FRAME, RECT)
            if len(EVM.frames) > 500:
                ys.append(int(np.mean(FOREHEAD[:, :, 1])))
                ys2 = np.array(ys)
                xs.append(time.perf_counter() - TIME_START)
                if len(EVM.frames) % fps == 0:
                    print(iter/fps, visualization_module.bpm(ys2, fps))
                    bpm = visualization_module.bpm(ys2, fps)

    cv2.imshow('Validation', FRAME)

    if key & 0xFF == ord('q'):
        break



CAP.release()
cv2.destroyAllWindows()
