import cv2 as cv
import numpy as np


def track_colour(lower_colour, upper_colour, camera_index=0, min_area=10, window_name='frame'):
    cap = cv.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, lower_colour, upper_colour)
            contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if len(contour) < 3 or cv.contourArea(contour) < min_area:
                    continue

                x, y, w, h = cv.boundingRect(contour)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                rect = cv.minAreaRect(contour)
                box = cv.boxPoints(rect).reshape(-1, 2).astype(np.int32)
                cv.drawContours(frame, [box], 0, (0, 0, 255), 2)

            cv.imshow(window_name, frame)
            if cv.waitKey(1) == ord('q'):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()
