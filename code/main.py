''' Live capture using webcam '''


import cv2
import roi_module


CAP = cv2.VideoCapture(0)
if not CAP.isOpened():
    raise IOError("Cannot open webcam")

while True:
    RET, FRAME = CAP.read()
    FRAME = cv2.resize(FRAME, None, fx=1, fy=1,
                       interpolation=cv2.INTER_AREA)

    cv2.imshow('Live face detection', FRAME)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

CAP.release()
cv2.destroyAllWindows()

roi_module.test()
