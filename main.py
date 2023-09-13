import pickle

import cv2
import face_recognition

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 414)

file = open('D:\PycharmProjects\openCv\EncodeFile.P', 'rb')
listWithId = pickle.load(file)
file.close()
encodeListKnown, Ids = listWithId
print(encodeListKnown)
print(Ids)

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(matches, faceDis)

    cv2.imshow("Display", img)
    cv2.waitKey(1)
