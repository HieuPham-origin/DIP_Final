import cv2
import numpy as np
import sys
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
cv2.imwrite('single.jpg', sample)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    return thresh
# im = cv2.imread('1.jpg')
# # cv2.imshow('cc', im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# im3 = im.copy()
# gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (5, 5), 0)
# thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

# # #################      Now finding Contours         ###################


# samples = np.empty((0, 100), np.float32)
# responses = []
# keys = [i for i in range(48, 58)]

# for cnt in contours:

#     if cv2.contourArea(cnt) > 100:
#         [x, y, w, h] = cv2.boundingRect(cnt)

#         if h > 45 and w >15:

#             cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
#             roi = thresh[y:y + h, x:x + w]
#             roismall = cv2.resize(roi, (10, 10))
#             cv2.imshow('norm', im)
#             key = cv2.waitKey(0)

#             if key == 27:  # (escape to quit)
#                 sys.exit()
#             elif key in keys:
#                 responses.append(int(chr(key)))
#                 sample = roismall.reshape((1, 100))
#                 samples = np.append(samples, sample, 0)
#             cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
# responses = np.array(responses, np.float32)
# responses = responses.reshape((responses.size, 1))
# print ("training complete")
# cv2.imwrite("train_result.png", im)
# np.savetxt('generalsamples.data', samples)
# np.savetxt('generalresponses.data', responses)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    return thresh
ar = []

# Thêm các ký tự từ '0' đến '9'
for i in range(10):
    ar.append(str(i))

# Thêm các ký tự từ 'A' đến 'Z'
for i in range(26):
    ar.append(chr(ord('A') + i))
    

# In danh sách kết quả
# print(ar)
import os
samples = np.empty((0, 100), np.float32)
responses = []

image_path =f'training_data'
for label in ar:  # Duyệt từ 0 đến 9
    label_folder = os.path.join(image_path, str(label))
    
    # Kiểm tra xem thư mục tồn tại hay không
    if os.path.exists(label_folder):
        # print(f"Processing label {label}: {label_folder}")
        
        # Duyệt qua các tệp trong thư mục
        for filename in os.listdir(label_folder):
            file_path = os.path.join(label_folder, filename)
            im = cv2.imread(file_path)

                # Thực hiện công việc cần thiết với mỗi tệp trong thư mục
            thresh =preprocess_image(im)
            # cv2.imshow('Cossntours', thresh)
            
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                    if cv2.contourArea(cnt) > 20:
                        [x, y, w, h] = cv2.boundingRect(cnt)

                        if h > 28 and w >25:
                            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            roi = thresh[y:y + h, x:x + w]
                            roismall = cv2.resize(roi, (10, 10))
                            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            responses.append(ord(label))
                            # print(ord(label))
                            sample = roismall.reshape((1, 100))
                            samples = np.append(samples, sample, 0)

                            # cv2.imshow('Contours', im)
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()
                            # responses.append(label)  # Gán nhãn tự động dựa trên số thư mục
                            # sample = roismall.reshape((1, 100))
                            # samples = np.append(samples, sample, 0)

                # Hiển thị hình ảnh với các đường viền
                
        
    else:
        print(f"Label {label} folder does not exist.")
np.savetxt('generalsamples.data', samples)
np.savetxt('generalresponses.data', responses)