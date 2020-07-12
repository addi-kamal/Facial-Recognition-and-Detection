# import libraries
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")
classifier = cv2.face.LBPHFaceRecognizer_create()
classifier.read("MyClassifier.yml")
image_id = 0
video = cv2.VideoCapture(0)

while True :
    _, img = video.read()
    img = cv2.flip(img, 1)  # flip the cam to act as a mirror
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect faces with the detectMultiscale function and store it in faces var
    faces = face_cascade.detectMultiScale(gray)

    # drawing rectangles around each fac
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (34,139,34), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        id, conf = classifier.predict(roi_gray)
        if conf <= 95:
            color = (34,139,34)
            name = "kamal"
        else:
            color = (100, 154, 244)
            name = "inconnu"
        label = name + " "+"{:5.2f}".format(conf)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,201,87), 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 100), 2)
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()