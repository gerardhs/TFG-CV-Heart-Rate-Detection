import heartpy as hp
import numpy as np
import cv2
from scipy import signal


class GRAPH():

    def __init__(self, width, height, max_values, type, fps, low, high):
        self.avg_color = []
        self.blue = []
        self.green = []
        self.red = []
        self.filtered_graph = []
        self.width = width
        self.height = height
        self.graph = np.zeros((self.height, self.width, 3), np.uint8)
        self.max_values = max_values
        self.type = type


    def bpm(self, ys, fs):
        working_data, measures = hp.process(ys, fs, high_precision=True)
        return measures['bpm']

    def refresh(self):
        self.graph = np.zeros((self.height, self.width, 3), np.uint8)
        if self.type == 'color':
            cv2.putText(self.graph, 'Magnified color intensity graph',
                        (int(self.width/3), 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA, False)
        elif self.type == 'bpm':
            cv2.putText(self.graph, 'RPPG signal graph',
                        (int(self.width/2.5), 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA, False)
        if len(self.filtered_graph) > self.max_values:
            self.filtered_graph.pop(0)
        if len(self.avg_color) > self.max_values:
            self.blue.pop(0)
            self.green.pop(0)
            self.red.pop(0)
            self.avg_color.pop(0)

    def draw_color_graph(self, data, color):
        cv2.putText(self.graph, '0', (1, 250), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA, False)
        cv2.putText(self.graph, '255', (1, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA, False)
        scale_factor_x = float(self.width) / self.max_values
        for i in range(0, len(data) - 1):
            curr_x = int(i * scale_factor_x)
            curr_y = 255 - int(data[i])
            next_x = int((i + 1) * scale_factor_x)
            next_y = 255 - int(data[i + 1])
            if color == 'blue':
                cv2.line(self.graph, (curr_x, curr_y),
                         (next_x, next_y), color=(255, 0, 0), thickness=1)
            elif color == 'green':
                cv2.line(self.graph, (curr_x, curr_y),
                         (next_x, next_y), color=(0, 255, 0), thickness=1)
            elif color == 'red':
                cv2.line(self.graph, (curr_x, curr_y),
                         (next_x, next_y), color=(0, 0, 255), thickness=1)
            elif color == 'white':
                cv2.line(self.graph, (curr_x, curr_y),
                         (next_x, next_y), color=(255, 255, 255), thickness=1)

    def draw_bpm_graph(self):
        scale_factor_x = float(self.width - 200) / self.max_values
        scale_factor_y =\
            (float(self.height)/2.0) / max(abs(np.array(self.filtered_graph)))
        for i in range(0, len(self.filtered_graph) - 1):
            curr_x = int(i * scale_factor_x)
            curr_y =\
                int(self.height/2 + self.filtered_graph[i]*scale_factor_y)
            next_x = int((i + 1) * scale_factor_x)
            next_y =\
                int(self.height/2 + self.filtered_graph[i + 1]*scale_factor_y)
            cv2.line(self.graph, (curr_x, curr_y),
                     (next_x, next_y), color=(0, 255, 0), thickness=1)

    def write_bpm(self, bpm, AVG_VALUES, FRAME_BUFFER_SIZE):
        cv2.putText(self.graph, '{} BPM'.format(bpm),
                    (int(self.width - 180), int(self.height/2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    1, cv2.LINE_AA, False)
        cv2.putText(self.graph, '{} / {} frames'.format(len(AVG_VALUES),
                                                        FRAME_BUFFER_SIZE),
                    (int(self.width - 180), int(self.height - 50)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                    1, cv2.LINE_AA, False)
