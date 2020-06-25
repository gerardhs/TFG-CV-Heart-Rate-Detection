''' Live capture using webcam '''


import time
import numpy as np
from scipy import signal
import cv2
import dlib
import roi_module
from evm_module import Eulerian_Video_Magnification
from visualization_module import GRAPH


EVM_FRAMES = 100
BPM_FRAME_BUFFER_SIZE = 500
BPM_MIN_FRAMES = 100
MIN_HZ = 0.83
MAX_HZ = 3.33


def bandpass_filter(data, low, high, sample_rate, order=4):
    nyquist_rate = sample_rate * 0.5
    low /= nyquist_rate
    high /= nyquist_rate
    b, a = signal.cheby2(order, 20, [low, high], btype='band')
    return signal.filtfilt(b, a, data)


def filter_signal_data(AVG_VALUES, sample_rate):
    x = [i for i in range(len(AVG_VALUES))]
    model = np.polyfit(x, AVG_VALUES, 6)
    predicted = np.polyval(model, x)
    values = np.array(AVG_VALUES)
    np.nan_to_num(values, copy=False)
    DETREND = values - predicted
    filtered = bandpass_filter(DETREND, MIN_HZ, MAX_HZ, sample_rate, order=4)
    return filtered


def get_bpm(DATA, old_bpm, sample_rate):
    fft = np.fft.rfft(DATA)
    freqs = np.fft.fftfreq(len(DATA), d=1/sample_rate)
    lower_bound = (np.abs(freqs - MIN_HZ)).argmin()
    higher_bound = (np.abs(freqs - MAX_HZ)).argmin()
    fft[:lower_bound] = 0
    fft[higher_bound:] = 0
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

EVM = Eulerian_Video_Magnification(lvl=6, amplification=2,
                                   frame_buffer_size=EVM_FRAMES)

BPM_GRAPH = GRAPH(890, 200, 100, 'bpm')

TIMES = []
AVG_VALUES = []
Y = []
FACES = []
BPMS = []
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
            BPM = 0

        for (i, FACE) in enumerate(FACES):

            EVM.frames.append(FRAME_ORIG)
            EVM.build_pyramids(FRAME_ORIG)
            EVM.times.append(time.time())

            roi_module.rectangles(i, FACE, FRAME)

            if len(EVM.pyramids) > EVM.frame_buffer_size:
                FRAME_MAGNIFIED = cv2.convertScaleAbs(EVM.magnify(FRAME_ORIG,
                                                                  MIN_HZ,
                                                                  MAX_HZ))

                MEAN_GREEN = roi_module.roi_extraction(PREDICTOR,
                                                       FRAME_MAGNIFIED, FRAME,
                                                       FACE)

                AVG_VALUES.append(MEAN_GREEN)
                TIMES.append(time.time())

                if len(BPMS) > 20:
                    BPM_GRAPH.write_bpm(int(np.array(BPMS[-10:]).mean()),
                                        AVG_VALUES, BPM_FRAME_BUFFER_SIZE)

                if len(AVG_VALUES) > BPM_FRAME_BUFFER_SIZE:
                    AVG_VALUES.pop(0)
                    TIMES.pop(0)

                if len(AVG_VALUES) > BPM_MIN_FRAMES:
                    TIME_ELAPSED = TIMES[-1] - TIMES[0]
                    SAMPLING_RATE = len(AVG_VALUES) / TIME_ELAPSED
                    FILTERED = filter_signal_data(AVG_VALUES, SAMPLING_RATE)
                    BPM_GRAPH.filtered_graph.append(FILTERED[-1])
                    BPM = get_bpm(FILTERED, BPM, SAMPLING_RATE)
                    BPMS.append(BPM)
                    BPM_GRAPH.draw_bpm_graph()
    else:
        EVM.frames = []
        EVM.pyramids = []
        AVG_VALUES = []
        TIMES = []
        FACES = []
        BPMS = []
        BPM = 0

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

    if len(EVM.frames) > EVM_FRAMES:
        cv2.putText(INFO_PANEL,
                    'EVM READY', (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA, False)

    else:
        cv2.putText(INFO_PANEL,
                    'EVM NOT READY', (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1, cv2.LINE_AA, False)
        cv2.putText(INFO_PANEL,
                    '{}/{}'.format(len(EVM.frames), EVM_FRAMES),
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

    if len(EVM.times) > EVM_FRAMES:
        EVM.frames.pop(0)
        EVM.times.pop(0)

    BPM_GRAPH.refresh()
    KEY = cv2.waitKey(1)

    if KEY == 32:
        START = not START
    if KEY & 0xFF == ord('q'):
        break

CAP.release()
cv2.destroyAllWindows()
