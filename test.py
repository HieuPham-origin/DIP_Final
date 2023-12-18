import cv2
import numpy as np

img_gray = cv2.imread('52100846.jpg')
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

half_len = img_gray.shape[0] // 2


bottom = img_gray[:half_len, :] 
height, width = bottom.shape[:2]
# cv2.imshow('fuaall', bottom)
print(width)
one_third_width = height // 2
sample = bottom[500:620, 650:1500]  

stacked_samples = [sample.copy() for _ in range(6)]

# Ghép các ảnh sample vào nhau theo chiều dọc
stacked_image = cv2.vconcat(stacked_samples)

# Hiển thị ảnh ghép
cv2.imwrite('Stacked.jpg', stacked_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
import numpy as np
import cv2
import sys

im = cv2.imread('1.jpg')
im3 = im.copy()

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

#################      Now finding Contours         ###################

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

samples = np.empty((0, 100), np.float32)
responses = []
keys = [i for i in range(48, 58)]

for cnt in contours:

    if cv2.contourArea(cnt) > 50:
        [x, y, w, h] = cv2.boundingRect(cnt)

        if h > 48:

            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi = thresh[y:y + h, x:x + w]
            roismall = cv2.resize(roi, (10, 10))
            cv2.imshow('norm', im)
            key = cv2.waitKey(0)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1, 100))
                samples = np.append(samples, sample, 0)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size, 1))
print ("training complete")
cv2.imwrite("train_result.png", im)
np.savetxt('generalsamples.data', samples)
np.savetxt('generalresponses.data', responses)
