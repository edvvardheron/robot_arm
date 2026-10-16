'''collect chessboard calibration data for phase three'''
import cv2 as cv
import numpy as np
import os
import time

def collect_chessboard_calibration_data(camera_index=0, chessboard_size=(9, 6), square_size=1.0, num_images=20, save_dir='checkerboard_calibration_images'):
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Prepare object points based on the chessboard size and square size
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    cap = cv.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    collected_images = 0
    try:
        while collected_images < num_images:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

            display = frame
            if ret:
                collected_images += 1
                image_path = os.path.join(save_dir, f'calibration_image_{collected_images}.png')
                cv.imwrite(image_path, frame)
                print(f"Saved {image_path}")
                display = frame.copy()
                cv.drawChessboardCorners(display, chessboard_size, corners, ret)
                time.sleep(3)  # Wait for a second before capturing the next image

            cv.imshow('Calibration', display)
            if cv.waitKey(1) == ord('q'):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    collect_chessboard_calibration_data()