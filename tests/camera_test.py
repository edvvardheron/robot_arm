import cv2 as cv
import numpy as np




def find_colour(lower_colour1, upper_colour1, lower_colour2, upper_colour2, camera_index=0, min_area=1000, window_name='frame'):
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
            # hsv is used to filter out the specific color range, hue saturation and value are used to define the color range
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mask1 = cv.inRange(hsv, lower_colour1, upper_colour1)
            mask2 = cv.inRange(hsv, lower_colour2, upper_colour2)
            mask = cv.bitwise_or(mask1, mask2)
            contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if len(contour) < 3 or cv.contourArea(contour) < min_area:
                    continue

                largest_contour = max(contours, key=cv.contourArea)
                x, y, w, h = cv.boundingRect(largest_contour)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                rect = cv.minAreaRect(contour)
                box = cv.boxPoints(rect).reshape(-1, 2).astype(np.int32)
                cv.drawContours(frame, [box], 0, (0, 0, 255), 2)
                #object coordinates
                object_x = x + w // 2
                object_y = y + h // 2
                print(f"object height {h} object width {w} area {cv.contourArea(largest_contour)}")
                #show blob center
                cv.circle(frame, (object_x, object_y), 5, (255, 0, 0), -1)

            cv.imshow(window_name, frame)
            if cv.waitKey(1) == ord('q'):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()




if __name__ == "__main__":
    lower_red1 = (0, 100, 100)
    upper_red1 = (5, 255, 255)
    lower_red2 = (170, 100, 100)
    upper_red2 = (179, 255, 255)
    find_colour(lower_red1, upper_red1, lower_red2, upper_red2)
