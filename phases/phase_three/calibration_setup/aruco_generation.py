'''generate aruco markers for the robot to detect and use for navigation'''

import cv2 as cv

def generate_aruco_markers():
    # Define the ArUco dictionary and parameters
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

    # Generate and save ArUco markers
    for i in range(6):
        marker_image = cv.aruco.generateImageMarker(aruco_dict, i, 200)
        cv.imwrite(f'aruco_marker_{i}.png', marker_image)
        print(f"Generated aruco_marker_{i}.png")

if __name__ == "__main__":
    generate_aruco_markers()