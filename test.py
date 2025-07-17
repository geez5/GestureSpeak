import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

counter = 0
folder = "ImagesData/A"

labels = ["A","B","C","D","E","F","I LOVE YOU"]

while True:
    
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        imgCropShape = imgCrop.shape

        aspectratio = h/w

        if aspectratio > 1:
            k = imgSize/h
            widthCal = math.ceil(k*w)
            imgResize = cv2.resize((imgCrop), (widthCal, imgSize))
            imgResizeShape = imgResize.shape
            widthGap = math.ceil((imgSize-widthCal)/2)
            imgWhite[:, widthGap:widthCal+widthGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction,index)


        else:
            k = imgSize/w
            heightCal = math.ceil(k*h)
            imgResize = cv2.resize((imgCrop), (imgSize, heightCal))
            imgResizeShape = imgResize.shape
            heightGap = math.ceil((imgSize-heightCal)/2)
            imgWhite[heightGap:heightCal+heightGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction,index)
        
        cv2.putText(imgOutput, labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),4)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,10,255),4)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("Imagewhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
