import cv2
import numpy as np
import os


results_path = os.path.join(os.getcwd(), 'results')
test_path = os.path.join(os.getcwd(), 'test')

image = cv2.imread(os.path.join(test_path, 'f1.bmp'))
cv2.imshow('original', cv2.resize(image, (800, 600)))

# (hMin = 0 , sMin = 0, vMin = 0), (hMax = 179 , sMax = 255, vMax = 83)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
lower_gray1 = np.array([0, 0, 0]) 
upper_gray1 = np.array([179, 255, 83]) 
mask1 = cv2.inRange(hsv, lower_gray1, upper_gray1)  
result1 = cv2.bitwise_and(image, image, mask = mask1) 
cv2.imshow('inner', cv2.resize(result1, (800, 600)))

# Dilation:
gray = cv2.cvtColor(result1,cv2.COLOR_HSV2BGR) 
gray = cv2.cvtColor(result1,cv2.COLOR_BGR2GRAY) 

kernel = np.ones((5,5), np.uint8)
img_dilation = cv2.dilate(result1, kernel, iterations=1)
cv2.imshow('Inner dilated image', cv2.resize(img_dilation, (800, 600)))

# Canny
edged = cv2.Canny(img_dilation, 100, 255)
cv2.imshow('Inner Canny Edges After Contouring', cv2.resize(edged, (800, 600)))


contours, hierarchy = cv2.findContours(edged,
    cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    
cv2.drawContours(image, contours, -1, (0, 0, 255), thickness = cv2.FILLED)
cv2.imshow('Inner Contours', cv2.resize(image, (800, 600)))


# (hMin = 0 , sMin = 0, vMin = 0), (hMax = 179 , sMax = 255, vMax = 166)
# (hMin = 0 , sMin = 0, vMin = 179), (hMax = 179 , sMax = 255, vMax = 255)
hsv2 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
lower_gray2 = np.array([0, 0, 179]) 
upper_gray2 = np.array([179, 255, 255]) 
mask2 = cv2.inRange(hsv, lower_gray2, upper_gray2)  
result2 = cv2.bitwise_and(image, image, mask = mask2) 
cv2.imshow('outer', cv2.resize(result2, (800, 600)))

gray2 = cv2.cvtColor(result2,cv2.COLOR_HSV2BGR) 
gray2 = cv2.cvtColor(result2,cv2.COLOR_BGR2GRAY) 

edged2 = cv2.Canny(gray2, 250, 255)
cv2.imshow('Outer Canny Edges After Contouring', cv2.resize(edged2, (800, 600)))

kernel2 = np.ones((5,5), np.uint8)
img_dilation2 = cv2.dilate(edged2, kernel, iterations=1)
cv2.imshow('Outer dilated image', cv2.resize(img_dilation2, (800, 600)))

contours2, hierarchy = cv2.findContours(img_dilation2,
    cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    
cv2.drawContours(image, contours2, -1, (0, 0, 255))#, thickness = cv2.FILLED)
cv2.imshow('Outer Contours', cv2.resize(image, (800, 600)))

# print(contours)
# print('=================================================================')
# print(contours2)
# print('=================================================================')

cv2.waitKey(0)
cv2.destroyAllWindows()