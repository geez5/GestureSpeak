import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
counter = 0

folder = "ImagesData/Z"

while True:
    
    success, img = cap.read()
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

        else:
            k = imgSize/w
            heightCal = math.ceil(k*h)
            imgResize = cv2.resize((imgCrop), (imgSize, heightCal))
            imgResizeShape = imgResize.shape
            heightGap = math.ceil((imgSize-heightCal)/2)
            imgWhite[heightGap:heightCal+heightGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("Imagewhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
