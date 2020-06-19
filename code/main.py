''' Live capture using webcam '''


import time
import numpy as np
from scipy import signal
import cv2
import dlib
import roi_module
from evm_module import Eulerian_Video_Magnification
from visualization_module import GRAPH

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

EVM_FRAMES = 100
BPM_FRAME_BUFFER_SIZE = 500
BPM_MIN_FRAMES = 100
MIN_HZ = 0.83
MAX_HZ = 3.33


def bandpass_filter(data, low, high, sample_rate, order=4):
    nyquist_rate = sample_rate * 0.5
    low /= nyquist_rate
    high /= nyquist_rate
    #b, a = signal.butter(order, [low, high], btype='band')
    #b, a = signal.bessel(order, [low, high], btype='band')
    b, a = signal.cheby2(order, 30, [low, high], btype='band')
    return signal.lfilter(b, a, data)


def filter_signal_data(AVG_VALUES, sample_rate):
    values = np.array(AVG_VALUES)
    np.nan_to_num(values, copy=False)
    DETREND = signal.detrend(values, type='linear')
    filtered = bandpass_filter(DETREND, MIN_HZ, MAX_HZ, sample_rate, order=2)
    return filtered


def get_bpm(DATA, frame_buffer_size, old_bpm, sample_rate):
    fft = np.fft.rfft(DATA)

    freqs = sample_rate / frame_buffer_size * np.arange(frame_buffer_size / 2 + 1)
    lower_bound = (np.abs(freqs - MIN_HZ)).argmin()
    higher_bound = (np.abs(freqs - MAX_HZ)).argmin()
    #fft[:lower_bound] = 0
    #fft[higher_bound:-higher_bound] = 0
    #fft[-lower_bound] = 0
    fft[:lower_bound] = 0
    fft[higher_bound:] = 0
    plotfft = np.copy(fft)
    bpm = freqs[np.argmax(fft)] * 60
    if old_bpm > 0:
        bpm = 0.95 * old_bpm + 0.05 * bpm
    return int(bpm)


CAP = cv2.VideoCapture(0)

if not CAP.isOpened():
    raise IOError("Cannot open webcam")

CAP.set(cv2.CAP_PROP_FPS, 30)
FPS = CAP.get(cv2.CAP_PROP_FPS)

PREDICTOR = dlib.shape_predictor('shape_predictor_81_face_landmarks.dat')
DETECTOR = dlib.get_frontal_face_detector()

EVM = Eulerian_Video_Magnification(lvl=6, amplification=100,
                                   frame_buffer_size=EVM_FRAMES,
                                   attenuation=1, fps=FPS)

BPM_GRAPH = GRAPH(890, 200, 50, 'bpm')

TIMES = []
AVG_VALUES = []
FACES = []
BPM = 0
START = False

while True:
    RET, FRAME = CAP.read()
    FRAME_ORIG = FRAME.copy()
    FRAME_ORIG = cv2.resize(FRAME_ORIG[:, :, :], (640, 512))

    if START:
        FACES = DETECTOR(FRAME, 0)
        if len(FACES) == 0:
            EVM.frames = []
            EVM.pyramids = []
            AVG_VALUES = []
            TIMES = []
        else:
            EVM.frames.append(FRAME_ORIG)
            EVM.build_pyramids(FRAME_ORIG)

        for (i, FACE) in enumerate(FACES):
            roi_module.rectangles(i, FACE, FRAME, BPM)
            if len(EVM.frames) > EVM.frame_buffer_size:
                FRAME_MAGNIFIED = cv2.convertScaleAbs(EVM.magnify(FRAME_ORIG,
                                                                  MIN_HZ,
                                                                  MAX_HZ))
                FOREHEAD =\
                    roi_module.feature_detection_forehead(PREDICTOR,
                                                          FRAME_MAGNIFIED,
                                                          FRAME, FACE)

                FRONTAL =\
                    roi_module.feature_detection_face(PREDICTOR,
                                                      FRAME_MAGNIFIED,
                                                      FRAME, FACE)

                AVG_VALUES.append((np.mean(FOREHEAD[:, :, 1]) +
                                   np.mean(FRONTAL[:, :, 1])) / 2)

                TIMES.append(time.time())
                BPM_GRAPH.write_bpm(BPM, AVG_VALUES, BPM_FRAME_BUFFER_SIZE)

                if len(AVG_VALUES) > BPM_FRAME_BUFFER_SIZE:
                    AVG_VALUES.pop(0)
                    TIMES.pop(0)

                if len(AVG_VALUES) > BPM_MIN_FRAMES:
                    TIME_ELAPSED = TIMES[-1] - TIMES[0]
                    SAMPLING_RATE = len(AVG_VALUES) / TIME_ELAPSED
                    FILTERED = filter_signal_data(AVG_VALUES, SAMPLING_RATE)
                    BPM_GRAPH.filtered_graph.append(FILTERED[-1])
                    BPM = get_bpm(FILTERED, len(FILTERED), BPM, SAMPLING_RATE)
                    BPM_GRAPH.draw_bpm_graph()

    else:
        EVM.frames = []
        EVM.pyramids = []
        AVG_VALUES = []
        TIMES = []
        FACES = []

    FRAME = cv2.resize(FRAME[:, :, :], (640, 512)) / 256
    INFO_PANEL = np.zeros((FRAME.shape[0], 250, 3))

    if len(FACES) == 0:
        cv2.putText(INFO_PANEL,
                    'FACE NOT DETECTED', (45, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1, cv2.LINE_AA, False)
    else:
        cv2.putText(INFO_PANEL,
                    'FACE DETECTED', (62, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA, False)

    if len(EVM.frames) > EVM.frame_buffer_size:
        cv2.putText(INFO_PANEL,
                    'EVM READY', (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA, False)

    else:
        cv2.putText(INFO_PANEL,
                    'EVM NOT READY', (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1, cv2.LINE_AA, False)
        cv2.putText(INFO_PANEL,
                    '{}/{}'.format(len(EVM.frames), EVM.frame_buffer_size),
                    (95, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                    1, cv2.LINE_AA, False)

    if len(AVG_VALUES) < BPM_MIN_FRAMES:
        cv2.putText(INFO_PANEL,
                    'HEART RATE NOT READY', (30, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
                    False)
        cv2.putText(INFO_PANEL,
                    '{}/{}'.format(len(AVG_VALUES), BPM_MIN_FRAMES), (95, 275),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
                    False)
    else:
        cv2.putText(INFO_PANEL,
                    'HEART RATE READY', (50, 250), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA, False)

    if START:
        cv2.putText(INFO_PANEL,
                    'IN PROGRESS', (70, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA, False)

        cv2.putText(INFO_PANEL,
                    'SPACEBAR TO STOP', (45, FRAME.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
                    False)

    else:
        cv2.putText(INFO_PANEL,
                    'SPACEBAR TO START', (45, FRAME.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
                    False)

    cv2.putText(INFO_PANEL,
                'HEART RATE DETECTION', (30, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 255), 1, cv2.LINE_AA, False)

    cv2.putText(INFO_PANEL,
                'Q TO QUIT', (80, FRAME.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
                False)

    cv2.imshow('Heart Rate Detector',
               np.vstack((np.hstack((FRAME, INFO_PANEL)), BPM_GRAPH.graph)))

    BPM_GRAPH.refresh()
    KEY = cv2.waitKey(1)

    if KEY == 32:
        START = not START
    if KEY & 0xFF == ord('q'):
        break

CAP.release()
cv2.destroyAllWindows()


'''
x = []
i = 0
for elem in AVG_VALUES:
    x.append(i)
    i+=1

data1 = pd.DataFrame(zip(x, AVG_VALUES), columns=['Frame', 'Green Pixel Intensity'])
data2 = pd.DataFrame(zip(x, DETREND), columns=['Frame', 'Green Pixel Intensity'])
data3 = pd.DataFrame(zip(x, FILTERED), columns=['Frame', 'Green Pixel Intensity'])

sns.set(style='darkgrid')
sns.set_context("paper")
sns.set(font_scale=1.25)
fig, axs = plt.subplots(ncols=2)
sns.lineplot(x='Frame', y='Green Pixel Intensity', data=data1, ax=axs[0], color='darkred')
sns.lineplot(x='Frame', y='Green Pixel Intensity', data=data2, ax=axs[1], color='darkred')
plt.setp(axs)
axs[0].set_title('Original PPG Signal')
axs[1].set_title('Detrended PPG Signal')
plt.show()

fig, axs = plt.subplots(ncols=2)
sns.lineplot(x='Frame', y='Green Pixel Intensity', data=data2, ax=axs[0], color='darkred')
sns.lineplot(x='Frame', y='Green Pixel Intensity', data=data3, ax=axs[1], color='darkred')
plt.setp(axs)
axs[0].set_title('Detrended PPG Signal')
axs[1].set_title('Filtered PPG Signal')
plt.show()



sns.set(style='darkgrid')
sns.set_context("paper")
sns.set(font_scale=1.15)

data4 = pd.DataFrame(zip(freqs, plotfft.real), columns=['Freq(Hz)', 'Magnitud'])


fig, ax = plt.subplots(1,1)
sns.lineplot(x='Freq(Hz)', y='Magnitud', data=data4, color='darkred')
plt.axvline(freqs[np.argmax(plotfft)], color='blue', linewidth=1, linestyle='dashed')
plt.title('Fast Fourier Transform of PPG signal')
xt = ax.get_xticks()
xt = np.delete(xt,[0,5])
xt=np.append(xt,freqs[np.argmax(plotfft)])

xtl=xt.tolist()
xtl[-1]=str(round(freqs[np.argmax(plotfft)], 3))+' Hz'
ax.set_xticks(xt)
ax.set_xticklabels(xtl)

plt.show()
'''
