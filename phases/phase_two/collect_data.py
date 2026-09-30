import cv2 as cv
import time

'''this file will be used to take images of each class of the dataset and store them in a folder for later use.
it uses the opencv library to capture images from the webcam and saves them in a specified directory.'''

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
image_counter = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        image_counter += 1

        #take a picture once per second and save it to the dataset folder
        image_name = f'datasets/car2/image_{image_counter}.jpg'
        cv.imwrite(image_name, frame)
        time.sleep(1)
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    cap.release()
    cv.destroyAllWindows()