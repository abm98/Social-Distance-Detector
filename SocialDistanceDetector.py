### will import required libraries
import numpy as np
import time
import cv2
import math
import imutils

labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")


print("Loading Machine Learning Model.....")
# configPath and weightsPath loaded in network
detector = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

print("Starting Camera or Playing Video....")
cap = cv2.VideoCapture('pedestrians.mp4')    ### (0) from webcam


while(cap.isOpened()):
    ret, image = cap.read()
    image = imutils.resize(image, width=800)
    (H, W) = image.shape[:2]
    ln = detector.getLayerNames()
    ln = [ln[i[0] - 1] for i in detector.getUnconnectedOutLayers()]
# Detect Object
    blob = cv2.dnn.blobFromImage(image, 1/255, (224, 224),swapRB=True, crop=False)
    detector.setInput(blob)
    start = time.time()
    layerOutputs = detector.forward(ln)
    end = time.time()
    print("Prediction time/frame : {:.4f} seconds".format(end - start))

    classIDs = []
    confidences = []
    boxes = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.75 and classID == 0:

                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
#Rectangle coordinates
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))


                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.6,0.4)
    ind = []
    for i in range(0,len(classIDs)):
        if(classIDs[i]==0):
            ind.append(i)
    a = []
    b = []

    if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                a.append(x)
                b.append(y)

    distance=[]
    nsd = []
    for i in range(0,len(a)-1):
        for k in range(1,len(a)):
            if(k==i):
                break
            else:
                x_dist = (a[k] - a[i])
                y_dist = (b[k] - b[i])

                d = math.sqrt(x_dist * x_dist + y_dist * y_dist)
                distance.append(d)
                if(d <=100):

                    nsd.append(i)
                    nsd.append(k)
                nsd = list(dict.fromkeys(nsd))
                print(nsd)
    color = (0, 0, 255)
    for i in nsd:
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "Too Close"
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
    color = (0, 255, 0)
    if len(idxs) > 0:
        for i in idxs.flatten():
            if (i in nsd):
                break
            else:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = 'Normal'
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

            text = "Social Distancing Violations: {}".format(len(nsd))
            cv2.putText(image, text, (10, image.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (56, 255, 255), 3)

    cv2.imshow("Social Distance Monitoring", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
