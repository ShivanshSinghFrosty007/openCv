import os
import pickle

import cv2
import face_recognition

folderPath = 'D:\PycharmProjects\openCv\Images'
modePathList = os.listdir(folderPath)
imgList = []
Ids = []
for path in modePathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    Ids.append(os.path.splitext(path)[0])

def findEncodings(imagesList):

    encodeList = []

    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

encodeListKnown = findEncodings(imgList)
listWithId = [encodeListKnown, Ids]
print("done")

file = open('D:\PycharmProjects\openCv\EncodeFile.P', 'wb')
pickle.dump(listWithId, file)
file.close()

