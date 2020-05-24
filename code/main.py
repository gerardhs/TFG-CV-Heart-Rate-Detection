''' Live capture using webcam '''


import cv2
import dlib
import roi_module
from evm_module import Eulerian_Video_Magnification
from visualization_module import GRAPH
import numpy as np
from scipy import signal
import time


EVM_FRAMES = 200
BPM_FRAME_BUFFER_SIZE = 500
BPM_MIN_FRAMES = 100
MIN_HZ = 0.83
MAX_HZ = 3.33


def butterworth_filter(data, low, high, sample_rate, order=5):
    nyquist_rate = sample_rate * 0.5
    low /= nyquist_rate
    high /= nyquist_rate
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.lfilter(b, a, data)


def demean(data, num_windows):
    window_size = int(round(len(data) / num_windows))
    output = np.zeros(data.shape)
    for i in range(0, len(data), window_size):
        if i + window_size > len(data):
            window_size = len(data) - i
        sliced = data[i:i+window_size]
        output[i:i+window_size] = sliced - np.mean(sliced)
    return output


def filter_signal_data(AVG_VALUES, fps):
    values = np.array(AVG_VALUES)
    np.nan_to_num(values, copy=False)
    DETREND = signal.detrend(values, type='linear')
    DEMEAN = demean(DETREND, 15)
    filtered = butterworth_filter(DEMEAN, MIN_HZ, MAX_HZ, fps, order=5)
    return filtered


def get_bpm(DATA, frame_buffer_size, old_bpm, fps):
    fft = np.fft.rfft(DATA)
    freqs = fps / frame_buffer_size * np.arange(frame_buffer_size / 2 + 1)
    lower_bound = (np.abs(freqs - MIN_HZ)).argmin()
    higher_bound = (np.abs(freqs - MAX_HZ)).argmin()
    fft[:lower_bound] = 0
    fft[higher_bound:-higher_bound] = 0
    fft[-lower_bound] = 0
    idx = fft.argmax()
    bpm = freqs[idx] * 60
    if old_bpm > 0:
        bpm = 0.9 * old_bpm + 0.1 * bpm
    return int(bpm)


CAP = cv2.VideoCapture(0)

if not CAP.isOpened():
    raise IOError("Cannot open webcam")

CAP.set(cv2.CAP_PROP_FPS, 30)
FPS = CAP.get(cv2.CAP_PROP_FPS)

PREDICTOR = dlib.shape_predictor('shape_predictor_81_face_landmarks.dat')
DETECTOR = dlib.get_frontal_face_detector()
EVM = Eulerian_Video_Magnification(lvl=6, amplification=2,
                                   frame_buffer_size=EVM_FRAMES,
                                   attenuation=0.3, fps=FPS)
COLOR_GRAPH = GRAPH(1280, 200, 50, 'color', FPS, MIN_HZ, MAX_HZ)
BPM_GRAPH = GRAPH(1280, 200, 50, 'bpm', FPS, MIN_HZ, MAX_HZ)
#CAP.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#CAP.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
bpm = 0
TIMES = []
AVG_VALUES = []
FILTERED = []
BPMS = []
while True:

    RET, FRAME = CAP.read()
    FRAME_ORIG = FRAME.copy()
    #FRAME_ORIG[:, :, :] = FRAME_ORIG[:, :, :] / 1.5
    FRAME_ORIG = cv2.resize(FRAME_ORIG[:, :, :], (640, 512))
    EVM.frames.append(FRAME_ORIG)
    EVM.build_pyramids(FRAME_ORIG)

    FACES = DETECTOR(FRAME, 0)
    if len(FACES) == 0:
        EVM.frames = []
        EVM.pyramids = []
        AVG_VALUES = []
        TIMES = []
    for (i, FACE) in enumerate(FACES):
        roi_module.rectangles(i, FACE, FRAME, bpm)
        if len(EVM.frames) > EVM.frame_buffer_size:
            FRAME_MAGNIFIED = cv2.convertScaleAbs(EVM.magnify(FRAME_ORIG,
                                                              MIN_HZ, MAX_HZ))
            FOREHEAD = roi_module.feature_detection_forehead(PREDICTOR,
                                                             FRAME_MAGNIFIED,
                                                             FRAME, FACE)

            FRONTAL = roi_module.feature_detection_face(PREDICTOR,
                                                        FRAME_MAGNIFIED,
                                                        FRAME, FACE)
            COLOR_GRAPH.blue.append((np.mean(FOREHEAD[:, :, 0]) +
                                     np.mean(FRONTAL[:, :, 0])) / 2)

            COLOR_GRAPH.green.append((np.mean(FOREHEAD[:, :, 1]) +
                                      np.mean(FRONTAL[:, :, 1])) / 2)

            COLOR_GRAPH.red.append((np.mean(FOREHEAD[:, :, 2]) +
                                    np.mean(FRONTAL[:, :, 2])) / 2)

            COLOR_GRAPH.avg_color.append((COLOR_GRAPH.blue[-1] +
                                          COLOR_GRAPH.green[-1] +
                                          COLOR_GRAPH.red[-1]) / 3)
            COLOR_GRAPH.draw_color_graph(COLOR_GRAPH.blue, 'blue')
            COLOR_GRAPH.draw_color_graph(COLOR_GRAPH.green, 'green')
            COLOR_GRAPH.draw_color_graph(COLOR_GRAPH.red, 'red')
            COLOR_GRAPH.draw_color_graph(COLOR_GRAPH.avg_color, 'white')
            AVG_VALUES.append(COLOR_GRAPH.green[-1])
            TIMES.append(time.time())
            if len(AVG_VALUES) > BPM_FRAME_BUFFER_SIZE:
                AVG_VALUES.pop(0)
                TIMES.pop(0)
            BPM_GRAPH.write_bpm(bpm, AVG_VALUES, BPM_FRAME_BUFFER_SIZE)
            if len(AVG_VALUES) > BPM_MIN_FRAMES:
                TIME_ELAPSED = TIMES[-1] - TIMES[0]
                SAMPLING_RATE = len(AVG_VALUES) / TIME_ELAPSED
                FILTERED = filter_signal_data(AVG_VALUES, SAMPLING_RATE)
                BPM_GRAPH.filtered_graph.append(FILTERED[-1])
                bpm = get_bpm(FILTERED, len(FILTERED), bpm, SAMPLING_RATE)
                BPMS.append(bpm)
                BPM_GRAPH.draw_bpm_graph()
    FRAME = cv2.resize(FRAME[:, :, :], (640, 512))
    if len(EVM.frames) > EVM.frame_buffer_size:
        cv2.imshow('Heart Beat Detector',
                   np.vstack((np.vstack((np.hstack((FRAME, FRAME_MAGNIFIED)),
                              COLOR_GRAPH.graph)), BPM_GRAPH.graph)))

    else:
        EVM_IMG = np.zeros(FRAME_ORIG.shape, np.uint8)
        cv2.putText(EVM_IMG,
                    'Computing EVM... {}/{}'.format(len(EVM.frames),
                                                    EVM.frame_buffer_size),
                    (160, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    1, cv2.LINE_AA, False)
        cv2.imshow('Heart Beat Detector',
                   np.vstack((np.vstack((np.hstack((FRAME, EVM_IMG)),
                              COLOR_GRAPH.graph)), BPM_GRAPH.graph)))
    COLOR_GRAPH.refresh()
    BPM_GRAPH.refresh()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

CAP.release()
cv2.destroyAllWindows()
