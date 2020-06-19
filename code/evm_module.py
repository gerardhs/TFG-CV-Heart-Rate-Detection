'''Eulerian video magnification module'''


import cv2
import numpy as np


class Eulerian_Video_Magnification():
    '''Live Eulerian Video Color Magnification'''

    def __init__(self, lvl, amplification,
                 frame_buffer_size, attenuation, fps):
        self.frames = []
        self.pyramids = []
        self.lvl = lvl
        self.amplification = amplification
        self.frame_buffer_size = frame_buffer_size
        self.attenuation = attenuation
        self.fps = fps

    def gaussian_pyramid(self, src):
        FRAME = np.copy(src)
        pyramids = [FRAME]
        for i in range(self.lvl):
            FRAME = cv2.pyrDown(FRAME)
            pyramids.append(FRAME)
        return pyramids

    def build_pyramids(self, src):
        self.pyramids.append(self.gaussian_pyramid(src)[-1])

    def ideal_bandpassing(self, src, low, high, axis=0):
        frames = np.asarray(src, dtype=np.float64)
        fft = np.fft.fft(frames, axis=axis)
        freqs = np.fft.fftfreq(frames.shape[0], d=1.0/self.fps)
        lower_bound = (np.abs(freqs - low)).argmin()
        higher_bound = (np.abs(freqs - high)).argmin()
        #fft[:lower_bound] = 0
        #fft[higher_bound:-higher_bound] = 0
        #fft[-lower_bound] = 0
        fft[:lower_bound] = 0
        fft[higher_bound:] = 0
        iff = np.fft.ifft(fft, axis=axis)
        return np.abs(iff)

    def amplify(self, src):
        output = src.copy()
        output[:, :, :, 0] *= self.amplification * self.attenuation
        output[:, :, :, 1] *= self.amplification
        output[:, :, :, 2] *= self.amplification * self.attenuation
        return output

    def reconnstruct(self, src, amp):
        for i in range(self.lvl):
            amp = cv2.pyrUp(amp)
        return src + amp

    def magnify(self, FRAME, low, high):
        filtered =\
            self.ideal_bandpassing(self.pyramids[-self.frame_buffer_size:],
                                   low, high)
        #cv2.imshow('Temporal Processing', np.hstack((FRAME /256, cv2.resize(filtered[-1], (FRAME.shape[1], FRAME.shape[0])))))
        amplified = self.amplify(filtered)
        output = self.reconnstruct(FRAME, amplified[-1])
        return output
