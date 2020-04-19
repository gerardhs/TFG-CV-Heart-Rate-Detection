''' Live capture using webcam '''

from collections import OrderedDict
import cv2


def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    """ Visualize each facial landmark """
    overlay = image.copy()
    output = image.copy()

    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32), (163, 38, 32),
                  (180, 42, 220)]

    for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
        (j, k) = FACIAL_LANDMARKS_IDXS[name]
        pts = shape[j:k]

        if name == "jaw":
            for idx in range(1, len(pts)):
                pt_a = tuple(pts[idx-1])
                pt_b = tuple(pts[idx])
                cv2.line(overlay, pt_a, pt_b, colors[i], 2)

        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)

        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        return output


CAP = cv2.VideoCapture(0)
FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
FACIAL_LANDMARKS_IDXS = OrderedDict([("mouth", (48, 68)),
                                     ("right_eyebrow", (17, 22)),
                                     ("left_eyebrow", (22, 27)),
                                     ("right_eye", (36, 42)),
                                     ("left_eye", (42, 48)),
                                     ("nose", (27, 35)),
                                     ("jaw", (0, 17))])
if not CAP.isOpened():
    raise IOError("Cannot open webcam")

while True:
    RET, FRAME = CAP.read()
    FRAME = cv2.resize(FRAME, None, fx=1, fy=1,
                       interpolation=cv2.INTER_AREA)

    GRAY = cv2.cvtColor(FRAME, cv2.COLOR_BGR2GRAY)
    FACES = FACE_CASCADE.detectMultiScale(GRAY, 1.1, 5)

    for (x, y, w, h) in FACES:
        cv2.rectangle(FRAME, (x, y), (x+w, y+h), (0, 255, 255), 2)

    cv2.imshow('Live face detection', FRAME)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

CAP.release()
cv2.destroyAllWindows()
