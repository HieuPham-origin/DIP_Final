import cv2
import numpy as np
input_image = cv2.imread("52100104.jpg")
kernel = np.ones((5,5),np.uint8)
hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
lower_orange = (0, 0, 0)  # Lower HSV values for star color
upper_orage = (179, 255,70)   # Upper HSV values for star color

mask = cv2.inRange(hsv_image, lower_orange, upper_orage)
ret, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)

# mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
cv2.imwrite("1.jpg",thresh)
cv2.waitKey(0) 

cv2.destroyAllWindows() 