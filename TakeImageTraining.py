import cv2
import numpy as np
input_image = cv2.imread("center.png")
kernel = np.ones((5,5),np.uint8)
hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
lower_orange = (0, 0, 0)  # Lower HSV values for star color
upper_orage = (179, 255,75)   # Upper HSV values for star color

mask = cv2.inRange(hsv_image, lower_orange, upper_orage)
ret, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)
# img_noise = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# img_noise = cv2.morphologyEx(img_noise, cv2.MORPH_CLOSE, kernel)
# mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
# cv2.imshow('ss', img_noise)
cv2.imwrite("center.jpg",thresh)
cv2.waitKey(0)
cv2.destroyAllWindows() 