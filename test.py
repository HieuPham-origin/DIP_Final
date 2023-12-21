


import cv2
import numpy as np

#######   training part    ###############
samples = np.loadtxt('generalsamples.data', np.float32)
responses = np.loadtxt('generalresponses.data', np.float32)
responses = responses.reshape((responses.size, 1))

model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

############################# testing part  #########################
# im2 = cv2.imread('2.jpg')
imSize = cv2.imread('52100796.jpg')
im = cv2.imread('3.jpg')
im3 = cv2.imread('52100846.jpg')

height, width = imSize.shape[:2]
im = cv2.resize(im, (width//2,height//2))
im3 = cv2.resize(im3, (width//2,height//2))

print(im.shape)
out = np.zeros(im.shape, np.uint8)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt) > 100  and cv2.contourArea(cnt) < 480:
        [x, y, w, h] = cv2.boundingRect(cnt)
        if h > 19 and h<30 and w >12:
            # cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            roi = thresh[y:y + h, x:x + w]
            roismall = cv2.resize(roi, (10, 10))
            roismall = roismall.reshape((1, 100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.findNearest(roismall, k=1)
            # cv2.rectangle(im3, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # string = str(chr(int(results[0][0])))
            # cv2.putText(out, string, (x, y + h), 0, 1, (0, 255, 0))
            
            if(int((results[0][0]))>47 and int((results[0][0]))<59):
                print(y)
                if (y>500 and x<450) == False:
                    cv2.rectangle(im3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                

                if y>=450 and y<475 and x<450:
                # int(chr(int((results[0][0]))))
                    string = str(chr(int(results[0][0])))
                    cv2.putText(out, string, (x, y + h), 0, 1, (0, 255, 0))


cv2.imshow('im', im3)
cv2.imshow('out', out)
# cv2.imwrite('result.jpg', im)
# cv2.imwrite('result_out.jpg', out)

cv2.waitKey(0)
