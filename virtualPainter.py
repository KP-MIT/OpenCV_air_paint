import cv2
import numpy as np
# import time # use to display fps if required
import os
# import hand tracking module.py
import HandTrackingModule as htm

################################ SET BRUSH AND ERASER THICKNESS ################################################
brushThickness = 15
eraserThickness = 50
################################################################################################################

# Assign header images containing folder's name as file path
folderPath = "Header"
myList = os.listdir(folderPath)

overlayList = []

# import all the images in empty overLayList.
for imPath in myList:
    srcPath = f'{folderPath}/{imPath}'
    # debugging point : print(srcPath) # check source path
    image = cv2.imread(srcPath)
    overlayList.append(image)

header = overlayList[0]
# debugging point : print(len(overlayList)) # check number of images imported

# variable to change brush color in RGB.
drawColor = (255, 0, 255)

# capture video in high resolution parameter 3 - width : 4 - height
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)

# initial cartesian position of detected hand
xp, yp = 0, 0

# create canvas for writing
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    success, img = cap.read()

    # flip image for more intuitive orientation while writing.
    img = cv2.flip(img, 1)

    # create hand detector and extract landmark list
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # extract current p1[x1,y1] (tip : index) and p2[x2, y2] (tip : middle)
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        fingers = detector.fingerUp()
        # debugging point : print(fingers) # check number of fingers detected up

        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            # debugging point : print("Selection Mode") # print to check if modes are working
            # if index and middle fingers are detected and y position in header
            # and depending on range of x select Brush/Eraser Color.
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255 ,0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
                    # eraser is actually black
                    # we actually write on canvas not on image.

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        if fingers[1] and fingers[2] == False:
            #print("Drawing Mode")
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    # Writing on image Approach 1 :
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)
    # Approach 2 : img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0) # Low Brightness - remove gray image.


    # setting initial Header image
    img[0:125, 0:1280] = header
    cv2.imshow("Image", img)

    #cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)