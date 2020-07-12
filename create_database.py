# import libraries
import cv2

# Turn on my video in the current directory
video = cv2.VideoCapture("MyVideo.mp4")
# Load haarcascade file
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")
id = 2020   # initilize id to identify image
while True:
    # take a frame from the video
    _, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect face in our frame
    face = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 4)
    for x, y, w, h in face:
        # store the face with extention .png, here we are going to extract juste the face with frame[y:y+h, x:x+w]
        cv2.imwrite("database/Kamal/img-{}.png".format(id), frame[y:y+h, x:x+w])
        #  draw boundary around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        id+=1
    key = cv2.waitKey(1)
    if key == ord("q"):   # press q to finish
        break
    cv2.imshow("video", frame)
video.release()
cv2.destroyAllWindows()