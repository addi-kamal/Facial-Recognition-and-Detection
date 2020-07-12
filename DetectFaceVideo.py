import numpy as np
import cv2

# Turn on webcam, 0 using default webcam in our pc or -1 to use an external camera
WebCam = cv2.VideoCapture(0)

while WebCam.isOpened() :
    _, img = WebCam.read() # take a frame
    img = cv2.flip(img, 1)  # flip the cam to act as a mirror
    # Load haarcascade files
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect faces with the detectMultiscale function and store it in faces var
    faces = face_cascade.detectMultiScale(gray)
    # drawing rectangles around each face
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (34,139,34), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        cv2.putText(img, "Face", (x + 2, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,201,87), 2)
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Destroying output window
cv2.destroyAllWindows()
