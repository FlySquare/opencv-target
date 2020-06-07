#Python v3, OpenCV v3.4.2

from cv2 import cv2
import numpy as np


videoCapture = cv2.VideoCapture(0,0)#yada0


ret, frame = videoCapture.read()
rows, cols = frame.shape[:2]


w = 200
h = 300
col = int((cols - w) / 2)
row = int((rows - h) / 2)
shiftWindow = (col, row, w, h)


roi = frame[row:row + h, col:col + w]
roiHsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

lowLimit = np.array((0., 60., 32.))
highLimit = np.array((255., 255., 255.))
mask = cv2.inRange(roiHsv, lowLimit, highLimit)


roiHist = cv2.calcHist([roiHsv], [0], mask, [180], [0, 180])
cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)


terminationCriteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS , 10, 1)


while True:
  
    ret , frame = videoCapture.read()

    
    frameHsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    backprojectedFrame = cv2.calcBackProject([frameHsv], [0], roiHist, [0, 180], 1)

    # karanlıkta kalan alanları maskeleyelim
    mask = cv2.inRange(frameHsv, lowLimit, highLimit)
    backprojectedFrame &= mask

    # mean shift algoritmasını başlatalım
    ret, shiftWindow = cv2.meanShift(backprojectedFrame, shiftWindow, terminationCriteria)

    #Col, row artık mean shift ile elde edilen alandır
    col, row = shiftWindow[:2]

    #Görüntü üzerinde tespit edilen alanı çizelim
    frame = cv2.rectangle(frame, (col, row), (col + w, row + h), (255,255,0), 4)

    cv2.imshow('Agirlikli Ortalama Oteleme - Mean Shift', frame)
    
    k = cv2.waitKey(60) & 0xff

videoCapture.release()
cv2.destroyAllshiftWindows()