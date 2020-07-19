import cv2
import numpy as np

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*perimeter, True)
            objCorners = len(approx)
            #Create boundy box
            x, y, w, h = cv2.boundingRect(approx)

            if objCorners == 3 : objectType = "T"
            elif objCorners == 4:
                aspectRatio = w/ float(h)
                if aspectRatio > 0.95 and aspectRatio < 1.05: objectType = "S"
                else: objectType = "R"
            elif objCorners > 4: objectType = "C"
            else: objectType = "X"
            cv2.rectangle(imgContour, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(imgContour, objectType,
                        (x + (w // 2) - 10, y + (h // 2) - 30), cv2.FONT_HERSHEY_COMPLEX, 0.4,
                        (255, 255, 255), 1)
#Read the image
path = 'Resources/img7.jpg'
img = cv2.imread(path)
#Resize the image, in case that image is too big
img = cv2.resize(img, (500, 300))

# Create a copy of the initial image
#imgBlank = np.zeros_like(img)

#Color detection
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

red_lower1 = np.array([0, 100, 70])
red_upper1 = np.array([10, 255, 255])

red_lower2 = np.array([170, 100, 70])
red_upper2 = np.array([180, 255, 255])

blue_lower = np.array([100, 150, 0])
blue_upper = np.array([140, 255, 255])

yellow_lower = np.array([15, 110, 110])
yellow_upper = np.array([25, 255, 255])

red1_mask = cv2.inRange(imgHSV, red_lower1, red_upper1)
red2_mask = cv2.inRange(imgHSV, red_lower2, red_upper2)
blue_mask = cv2.inRange(imgHSV, blue_lower, blue_upper)
yellow_mask = cv2.inRange(imgHSV, yellow_lower, yellow_upper)

final_mask = red1_mask + red2_mask + blue_mask + yellow_mask
imgColor = cv2.bitwise_and(img, img, mask=final_mask)

#Convert to gray scale and blury
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7,7), 0)

#Detec the edges
#imgLaplacian = cv2.Laplacian(imgGray, cv2.CV_64F, 3)
imgCanny = cv2.Canny(imgBlur, 50, 50)

#Shape detection
imgContour = imgColor.copy()
getContours(imgCanny)

# Showing all the images
imgStack = stackImages(0.8, ([img, imgBlur, imgCanny],
                             [imgHSV, imgColor, imgContour]))
cv2.imshow("Stack", imgStack)
cv2.waitKey(0)