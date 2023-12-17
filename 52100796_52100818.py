import numpy as np
import cv2 
import numpy as np
class DetectStudentNumber:
    def __init__(self, image_path, train_img):
        self.image_path = image_path
        self.train_img = train_img
    def preprocessing(self):
        return None
    def boundingBox(self):
        return None
# arr1 = np.array([223, 225, 233])
# arr2 = np.array([123, 456, 789])

# print(np.all(arr1 == arr2))
image = cv2.imread('52100018.jpg')

# print(image.shape)
# shape = image.shape
# (w,h) = shape[:2]
# h_new = int(h/2)
# w_new = int(w/2)
# print(h, w)
# image = cv2.resize(image,(h_new, w_new))
detect = cv2.barcode.BarcodeDetector()
retval, decoded_info, decoded_type, points = detect.detectAndDecodeWithType(image)
print(retval)
print(decoded_info)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# cv2.imshow('hsv', image)
# # Define a threshold range around the HSV color
# hue_threshold = 30  # Adjust this value as needed
# saturation_threshold = 60  # Adjust this value as needed
# value_threshold = 30  # Adjust this value as needed
# # Calculate lower and upper threshold values
# def range(hue,sa,va):
#     lower_hue = max(0, hue - hue_threshold)
#     upper_hue = min(179, hue + hue_threshold)
#     lower_saturation = max(0, sa - saturation_threshold)
#     upper_saturation = min(255, sa + saturation_threshold)
#     lower_value = max(0, va - value_threshold)
#     upper_value = min(255, va + value_threshold)
#     lower_skin = np.array([lower_hue, lower_saturation, lower_value], dtype=np.uint8)
#     upper_skin = np.array([upper_hue, upper_saturation, upper_value], dtype=np.uint8)
#     return upper_skin, lower_skin
# print(image[646][166])
# for i in range(len(image)):
#     for j in range(len(image[0])):
#         if np.all(image[i][j] == image[219][615]) or np.all(image[i][j] == image[334][379]):
#             image[i][j] = image[165][752]
#         else:
#             continue

# # cv2.imwrite("test_resized.jpg", image)
# # image = np.uint8(image)
# # image_2d = image.reshape((-1, image.shape[-1]))
# # print(image_2d)
# # np.savetxt('generalsamples.data', image_2d)
# cv2.imshow('rgb', image)

# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # ret, thresh1 = cv2.threshold(image, 105, 255, cv2.THRESH_BINARY)# + 
# #                                             # cv2.THRESH_OTSU)
# # th1 = cv2.adaptiveThreshold(thresh1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
# # # kernel = np.array([[0, 1, 0],
# # #                     [1, 1, 1],
# # #                     [0, 1, 0]], dtype = np.uint8)
# # # img_noise = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
# # # img_noise = cv2.morphologyEx(img_noise, cv2.MORPH_CLOSE, kernel)
# # contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # digit_counter = 0
# # for cnt in contours:
# #     if len(cnt) > 5 and cv2.contourArea(cnt) > 150:  
# #         x, y, w, h = cv2.boundingRect(cnt)
        
# #         if w / h < 2:
# #             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
# #             digit_counter += 1

# # cv2.imshow('Otsu Threshold', thresh1)
# # cv2.imshow('adaptive Threshold', th1)
# # cv2.imshow("huhu",image)

# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

