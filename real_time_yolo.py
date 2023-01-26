import cv2
import numpy as np
import time
import os
import face_recognition
import pandas as pd

# Load Yolo
net = cv2.dnn.readNet("weights/yolov3.weights", "cfg/yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
print(type(net.getUnconnectedOutLayers()))
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

path = 'faces'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def getEncodings():
    df=pd.read_csv("encodings.csv")
    return list(df[df.columns[1]])

encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Loading image
cap = cv2.VideoCapture("walk1.mp4")

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
while True:
    _, frame = cap.read()
    frame_id += 1

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)
    crop_img=[]
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            if label in "person":
                imgS=frame[y:y+h, x:x+w].copy()
                imgS2 = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

                facesCurFrame = face_recognition.face_locations(imgS2)

                encodesCurFrame = face_recognition.face_encodings(imgS2, facesCurFrame)

                for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                    # print(faceDis)
                    matchIndex = np.argmin(faceDis)

                    if matches[matchIndex]:
                        name = classNames[matchIndex].upper()
                        # print(name)
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1, x2, y2, x1
                        cv2.rectangle(frame, (x+x1, y+y1), (x+x2, y+y2), (0, 255, 0), 2)

                        cv2.putText(frame, name, (x+x1 + 6, y+y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255),
                                    2)
                    else:
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1, x2, y2, x1
                        cv2.rectangle(frame, (x+x1, y+y1), (x+x2, y+y2), (0, 0, 255), 2)

                print(h)
                color=colors[0] if h>400 else colors[1]

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, color, 3)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 4, (0, 0, 0), 3)
    cv2.imshow("Image", frame)
    # for i in range(len(crop_img)):
    #     imgS = cv2.cvtColor(crop_img[i], cv2.COLOR_BGR2RGB)
    #
    #     facesCurFrame = face_recognition.face_locations(imgS)
    #
    #     encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    #
    #     for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
    #         matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
    #         faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
    #         # print(faceDis)
    #         matchIndex = np.argmin(faceDis)
    #
    #         if matches[matchIndex]:
    #             name = classNames[matchIndex].upper()
    #             # print(name)
    #             y1, x2, y2, x1 = faceLoc
    #             y1, x2, y2, x1 = y1, x2, y2, x1
    #             cv2.rectangle(crop_img[i], (x1, y1), (x2, y2), (0, 255, 0), 2)
    #
    #             cv2.putText(crop_img[i], name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    #         else:
    #             y1, x2, y2, x1 = faceLoc
    #             y1, x2, y2, x1 = y1, x2, y2, x1
    #             cv2.rectangle(crop_img[i], (x1, y1), (x2, y2), (0, 0, 255), 2)
    #     cv2.imshow("image"+str(i),crop_img[i])
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()