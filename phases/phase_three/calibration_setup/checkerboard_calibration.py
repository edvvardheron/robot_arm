"""After capturing images of a chessboard pattern, this script calibrates the camera
and saves the calibration parameters to a YAML file.
"""

import cv2 as cv
import numpy as np
import yaml
import glob
import math

CHECKERBOARD_SIZE = (9, 6)  # Number of inner corners per a chessboard row and column
SQUARE_SIZE = 2.7  # Size of a square in mm

def calibrate_camera(image_dir='checkerboard_calibration_images', checkerboard_size=CHECKERBOARD_SIZE, square_size=SQUARE_SIZE, output_file='camera_calibration.yaml'):
    # Prepare object points based on the checkerboard size and square size
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store object points and image points from all images
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane
    used_filenames = []

    # Get all calibration images
    images = glob.glob(f'{image_dir}/*.png')

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv.findChessboardCorners(gray, checkerboard_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            used_filenames.append(fname)

            # Draw and display the corners
            cv.drawChessboardCorners(img, checkerboard_size, corners, ret)
            cv.imshow('img', img)
            cv.waitKey(500)
            print(f"Processed {fname}: Found corners.")
        else:
            print(f"Processed {fname}: Corners not found.")

    cv.destroyAllWindows()

    # Calibrate the camera
    rms_error, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print(f"Overall RMS reprojection error: {rms_error:.3f} px")

    # Per-image reprojection error, worst first, to help spot bad captures
    per_image_errors = []
    for i, fname in enumerate(used_filenames):
        projected, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i].reshape(-1, 2), projected.reshape(-1, 2), cv.NORM_L2) / math.sqrt(len(projected))
        per_image_errors.append((fname, error))

    for fname, error in sorted(per_image_errors, key=lambda x: -x[1]):
        print(f"  {fname}: {error:.3f} px")

    # Save the calibration parameters to a YAML file
    calibration_data = {
        'camera_matrix': mtx.tolist(),
        'distortion_coefficients': dist.tolist(),
        'rotation_vectors': [rvec.tolist() for rvec in rvecs],
        'translation_vectors': [tvec.tolist() for tvec in tvecs]
    }

    with open(output_file, 'w') as f:
        yaml.dump(calibration_data, f)

    print(f"Calibration parameters saved to {output_file}")

if __name__ == "__main__":
    calibrate_camera()
