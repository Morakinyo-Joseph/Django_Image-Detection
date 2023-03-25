from django.http import StreamingHttpResponse
from django.views.decorators import gzip
import cv2
import numpy as np


# # Define the lower and upper bounds of the ball color in HSV color space
# ball_color_lower = np.array([20, 100, 100])
# ball_color_upper = np.array([30, 255, 255])


# # Load the cascade classifier for detecting players
# player_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')


# # Get the video stream from the camera
# cap = cv2.VideoCapture(0)


# # Define a function to get the video frames and detect the ball movement
# def get_frame():
#     # Initialize the ball position
#     ball_position = None

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert the frame to HSV color space
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#         # Threshold the frame to get the binary image of the ball
#         ball_mask = cv2.inRange(hsv, ball_color_lower, ball_color_upper)

#         # Remove the noise from the ball mask using morphological operations
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#         ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)
#         ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, kernel)

#         # Find the contours of the ball in the binary image
#         contours, hierarchy = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # If a ball contour is found, get its position and draw a circle around it
#         if len(contours) > 0:
#             ball_contour = max(contours, key=cv2.contourArea)
#             ((x, y), radius) = cv2.minEnclosingCircle(ball_contour)
#             ball_position = (int(x), int(y))
#             cv2.circle(frame, ball_position, int(radius), (0, 255, 255), 2)

#         # Detect players in the frame and draw rectangles around them
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         players = player_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
#         for (x, y, w, h) in players:
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

#         # If the ball is detected and its position has changed, print the new position
#         if ball_position is not None:
#             cv2.putText(frame, f"Ball position: {ball_position}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
#             print(f"Ball position: {ball_position}")

#         # Encode the frame as a JPEG image
#         frame = cv2.resize(frame, (640, 480))
#         _, jpeg = cv2.imencode('.jpg', frame)
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


# # Decorate the function with gzip compression
# @gzip.gzip_page
# def video_feed(request):
#     return StreamingHttpResponse(get_frame(), content_type='multipart/x-mixed-replace; boundary=frame')


# Define the lower and upper bounds of the ball color in HSV color space
ball_color_lower = np.array([20, 100, 100])
ball_color_upper = np.array([30, 255, 255])


# Load the cascade classifier for detecting players
player_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')


# Get the video stream from the Android phone camera server
cap = cv2.VideoCapture('http://192.168.8.100:8080/video')


# Define a function to get the video frames and detect the ball movement
def get_frame():
    # Initialize the ball position
    ball_position = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the frame to get the binary image of the ball
        ball_mask = cv2.inRange(hsv, ball_color_lower, ball_color_upper)

        # Remove the noise from the ball mask using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, kernel)

        # Find the contours of the ball in the binary image
        contours, hierarchy = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If a ball contour is found, get its position and draw a circle around it
        if len(contours) > 0:
            ball_contour = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(ball_contour)
            ball_position = (int(x), int(y))
            cv2.circle(frame, ball_position, int(radius), (0, 255, 255), 2)

        # Detect players in the frame and draw rectangles around them
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        players = player_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in players:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # If the ball is detected and its position has changed, print the new position
        if ball_position is not None:
            cv2.putText(frame, f"Ball position: {ball_position}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            print(f"Ball position: {ball_position}")

        # Encode the frame as a JPEG image
        frame = cv2.resize(frame, (640, 480))
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


# Decorate the function with
@gzip.gzip_page
def video_feed(request):
    return StreamingHttpResponse(get_frame(), content_type='multipart/x-mixed-replace; boundary=frame')