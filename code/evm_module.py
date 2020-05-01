'''Eulerian video magnification module'''


import cv2
import numpy as np


class Eulerian_Video_Magnification():

    def __init__(self, lvl, amplification, fps, frame_buffer_size):
        self.frames = []
        self.pyramids = []
        self.lvl = lvl
        self.amplification = amplification
        self.fps = fps
        self.frame_buffer = frame_buffer_size

    def gaussian_pyramid(self, src):
        FRAME = src.copy()
        pyramid = [FRAME]
        for i in range(self.lvl):
            FRAME = cv2.pyrDown(FRAME)
            pyramid.append(FRAME)
        return pyramid

    def build_pyramids(self, src):
        self.pyramids.append(self.gaussian_pyramid(src)[-1])

    def amplify(self, src):
        return src * self.amplification

    def temporal_filter(self, src, low, high, axis=0):
        return ""

    def reconnstruct(self, src):
        for i in range(self.lvl):
            src = cv2.pyrUp(src)
        return src

    def magnify(self, FRAME, low, high):
        return ""
