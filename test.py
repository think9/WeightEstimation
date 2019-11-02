"""
Issue

1. 임계치 계산 자동화
-> 히스토그램 분석 시 임계치를 자동으로 탐색하여 정할 수 있도록한다.
-> Local Maximum을 이용?
"""

def getLineX(img):
    check = True
    x_start = 0
    x_end = 0
    y_ = 0
    x1 = 0
    x2 = 0
    
    len_y = img.shape[0]
    len_x = img.shape[1]

    for y in range(len_y):
        for x in range(len_x):
            if img[y][x] == 255:
                if check:
                    x_start = x
                    check = False
                else:
                    x_end = x
                    check = True
                    if x_end - x_start > x2 - x1:
                        y_ = y
                        x2 = x_end
                        x1 = x_start
    
    return x1, x2, y_

def getLineY(img):
    check = True
    y_start = 0
    y_end = 0
    x_ = 0
    y1 = 0
    y2 = 0
    
    len_y = img.shape[0]
    len_x = img.shape[1]

    for x in range(len_x):
        for y in range(len_y):
            if img[y][x] == 255:
                if check:
                    y_start = y
                    check = False
                else:
                    y_end = y
                    check = True
                    if y_end - y_start > y2 - y1:
                        x_ = x
                        y2 = y_end
                        y1 = y_start
    
    return y1, y2, x_

import cv2
import numpy as np
from matplotlib import pyplot as plt

#Read iamge and smoothing
img = cv2.imread('./sample.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_gauss = cv2.GaussianBlur(img_gray, (5, 5), 0)

#Histogram
"""
hist = cv2.calcHist(images = [frame_gauss], channels = [0], mask = None, histSize = [256], ranges = [0, 256])
hist = hist.flatten()
plt.title('hist')
plt.plot(hist, color = 'r')
binX = np.arange(256)
plt.bar(binX, hist, width = 1, color = 'b')
plt.show()
"""

#Make binary
ret, img_result = cv2.threshold(img_gauss, 140, 255, cv2.THRESH_BINARY)
result = cv2.bitwise_not(img_result)

#Edge detection
canny = cv2.Canny(result, 200, 255)

#가로, 세로 분석
len_y = img.shape[0]
len_x = img.shape[1]

x1, x2, y = getLineX(canny)
cv2.line(img, (x1, y), (x2, y), (255, 0, 0), 3)
x_length = round(25 * (x2 - x1) / len_x, 2)
x_text = "x : " + str(x_length) + "cm"
cv2.putText(img, x_text , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

y1, y2, x = getLineY(canny)
cv2.line(img, (x, y1), (x, y2), (0, 0, 255), 3)
y_length = round(25 * (y2 - y1) / len_y, 2)
y_text = "y : " + str(y_length) + "cm"
cv2.putText(img, y_text , (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

#부피 계산 - 1/3 * pi * r^2
volume = round(np.power(x_length / 2, 2) * np.pi / 3, 2)
v_text = "volume : " + str(volume) + "cm^3"
cv2.putText(img, v_text , (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

#모래 밀도 = 1.331g/cm^3
density = round(volume * 1.331, 2)
d_text = "weight : " + str(density) + "g"
cv2.putText(img, d_text , (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

cv2.imshow('result', img)

key = cv2.waitKey(0)
cv2.destroyAllWindows()