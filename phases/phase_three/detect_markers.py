'''detect aruco markers in the image and return their ids and corners, taking into account the camera calibration parameters'''

import cv2 as cv
import numpy as np
import yaml

with open('camera_calibration.yaml', 'r') as file:
    calibration_data = yaml.safe_load(file)

camera_matrix = np.array(calibration_data['camera_matrix'])
distortion_coefficients = np.array(calibration_data['distortion_coefficients'])
marker_length = 0.05  # Length of the marker's side in meters

aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
parameters = cv.aruco.DetectorParameters()

# create a detector instance with default parameters
detector = cv.aruco.ArucoDetector(aruco_dict, parameters)


cap = cv.VideoCapture(0)
num_markers_detected = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None and len(ids) > 0:
        num_markers_detected += len(ids)
        # Draw detected markers
        cv.aruco.drawDetectedMarkers(frame, corners, ids)

        # define 3d coords of the marker corners in the marker's coordinate system
        marker_corners_3d = np.array([[-marker_length / 2, marker_length / 2, 0],
                                      [marker_length / 2, marker_length / 2, 0],
                                      [marker_length / 2, -marker_length / 2, 0],
                                      [-marker_length / 2, -marker_length / 2, 0]], dtype=np.float32)

        for marker_corners, id in zip(corners, ids.flatten()):
            image_points = marker_corners.reshape(-1, 2).astype(np.float32)

            retval, rvec, tvec = cv.solvePnP(marker_corners_3d, image_points, camera_matrix, distortion_coefficients)

            if retval:
                # Draw the axis for the marker
                cv.drawFrameAxes(frame, camera_matrix, distortion_coefficients, rvec, tvec, 0.03)

                print(tvec) # Print the translation vector (position of the marker in camera coordinates)

                x, y, z = tvec.flatten()
                distance = np.linalg.norm(tvec)

                #print to terminal
                print(f"ID: {id}, Position: ({x:.2f}, {y:.2f}, {z:.2f}), Distance: {distance:.2f} m")

                #display the position and distance on the image
                org = (20, 30 + 30 * id)  # Adjust the y-coordinate based on the marker ID
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                color = (0, 255, 0)
                thickness = 2
                cv.putText(frame, f"ID: {ids.flatten()[0]}, Pos: ({x:.2f}, {y:.2f}, {z:.2f}), Dist: {distance:.2f} m", org, font, font_scale, color, thickness, cv.LINE_AA)



    cv.imshow('Aruco Marker Detection', frame)
    if cv.waitKey(1) == ord('q'):
        break