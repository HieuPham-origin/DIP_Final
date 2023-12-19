import cv2 
import numpy as np
import sys
img_gray = cv2.imread('52100846.jpg')
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

half_len = img_gray.shape[0] // 2
 
 

bottom = img_gray[:half_len, :] 
height, width = bottom.shape[:2]
# cv2.imshow('fuaall', bottom)
# print(width)
one_third_width = height // 2
sample = img_gray[929:991, 481:813]  

stacked_samples = [sample.copy() for _ in range(6)]

# Ghép các ảnh sample vào nhau theo chiều dọc
stacked_image = cv2.vconcat(stacked_samples)

cv2.imshow('ss', stacked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('Stacked2.jpg', stacked_image)