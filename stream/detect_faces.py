from django.http import StreamingHttpResponse
from django.views.decorators import gzip
import cv2
import os


'''this part captures all the images and renames it with an increment'''

# Load the face detection cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Get the video stream from your phone's camera
cap = cv2.VideoCapture('http:/192.168.8.100:8080/video')

# Create the media directory if it doesn't exist
if not os.path.exists('media'):
    os.makedirs('media')


# Define a function to get the video frames and detect faces
def get_frame():
    # Initialize the filename increment to 0
    filename_increment = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        # Draw a rectangle around each detected face
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            
            # Crop the face from the frame
            face = frame[y:y+h, x:x+w]
            
            # Construct the filename for the captured face
            filename = f"face_{filename_increment}.jpg"
            
            # Save the face as an image
            cv2.imwrite(os.path.join('media', filename), face)
            print(f"Face has being captured ({filename_increment})")
            
            # Increment the filename increment
            filename_increment += 1
        
        # Encode the frame as a JPEG image
        frame = cv2.resize(frame, (640, 480))
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

# Decorate the function with gzip compression
@gzip.gzip_page
def video_feed(request):
    return StreamingHttpResponse(get_frame(), content_type='multipart/x-mixed-replace; boundary=frame')


