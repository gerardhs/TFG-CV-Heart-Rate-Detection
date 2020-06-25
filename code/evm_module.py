'''Eulerian video magnification module'''


import cv2
import numpy as np
import matplotlib as plt


class Eulerian_Video_Magnification():
    '''Live Eulerian Video Color Magnification'''

    def __init__(self, lvl, amplification, frame_buffer_size):

        self.lvl = lvl
        self.amplification = amplification
        self.frame_buffer_size = frame_buffer_size
        self.frames = []
        self.times = []
        self.pyramids = []

    def gaussian_pyramid(self, src):
        FRAME = np.copy(src)
        pyramids = [FRAME]
        for i in range(self.lvl):
            FRAME = cv2.pyrDown(FRAME)
            pyramids.append(FRAME)
        return pyramids

    def build_pyramids(self, src):
        self.pyramids.append(self.gaussian_pyramid(src)[-1])

    def ideal_bandpassing(self, src, low, high):
        TIME_ELAPSED = self.times[-1] - self.times[0]
        SAMPLING_RATE = len(self.frames) / TIME_ELAPSED
        timestep = 1 / SAMPLING_RATE
        frames = np.asarray(src)
        fft = np.fft.fft(frames, axis=0)
        freqs = np.fft.fftfreq(len(frames), d=timestep)
        lower_bound = (np.abs(freqs - low)).argmin()
        higher_bound = (np.abs(freqs - high)).argmin()
        fft[:lower_bound] = 0
        fft[higher_bound:] = 0
        iff = np.fft.ifft(fft, axis=0)
        return np.abs(iff)

    def amplify(self, src):
        output = src.copy()
        output[:, :, :, 0] *= self.amplification
        output[:, :, :, 1] *= self.amplification
        output[:, :, :, 2] *= self.amplification
        return output

    def reconnstruct(self, src, amp):
        for i in range(self.lvl):
            amp = cv2.pyrUp(amp)
        return src + amp

    def magnify(self, FRAME, low, high):
        filtered =\
            self.ideal_bandpassing(self.pyramids[-self.frame_buffer_size:],
                                   low, high)
        amplified = self.amplify(filtered)
        output = self.reconnstruct(FRAME, amplified[-1])
        return output
