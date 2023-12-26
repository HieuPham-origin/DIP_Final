import cv2
import numpy as np
import os

class CreateTrainingSet:
    def __init__(self, data_path):
        self.data_path = data_path
        self.samples = None
        self.responses = None
    def preprocessing_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
        return thresh
    def create_label(self):
        ar = []
        for i in range(10):
            ar.append(str(i))
        # Thêm các ký tự từ 'A' đến 'Z'
        for i in range(26):
            ar.append(chr(ord('A') + i))
        return ar
    def training(self):
        self.samples = np.empty((0, 100), np.float32)
        self.responses = []
        ar = self.create_label()
        for label in ar:  # Duyệt từ 0 đến 9
            label_folder = os.path.join(self.data_path, str(label))
            # Kiểm tra xem thư mục tồn tại hay không
            if os.path.exists(label_folder):
                # Duyệt qua các tệp trong thư mục
                for filename in os.listdir(label_folder):
                    file_path = os.path.join(label_folder, filename)
                    im = cv2.imread(file_path)
                    # Thực hiện công việc cần thiết với mỗi tệp trong thư mục
                    thresh =self.preprocessing_image(im)
                    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                            if cv2.contourArea(cnt) > 20:
                                [x, y, w, h] = cv2.boundingRect(cnt)
                                if h > 28 and w >25:
                                    cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                                    roi = thresh[y:y + h, x:x + w]
                                    roismall = cv2.resize(roi, (10, 10))
                                    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
                                    self.responses.append(ord(label))
                                    # print(ord(label))
                                    sample = roismall.reshape((1, 100))
                                    self.samples = np.append(self.samples, sample, 0)
            else:
                print(f"Label {label} folder does not exist.")
        np.savetxt('generalsamples.data', self.samples)
        np.savetxt('generalresponses.data', self.responses)
        return "Ok"
    
## training complete so I comment this code, run when you still not train model.
# data_path = f'training_data'
# train = CreateTrainingSet(data_path)
# train.training()

class CheckWithKNN:
    def __init__(self, samples, responses):
        self.samples = samples
        self.responses = responses
        self.model = None
    def trainModel(self):
        self.model = cv2.ml.KNearest_create()
        self.model.train(self.samples, cv2.ml.ROW_SAMPLE, self.responses)
    def preprocess_image(self, image_path, value):
        input_image = cv2.imread(image_path)
        hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
        lower_range = (0, 0, 0)  # Lower HSV values
        upper_range = (179, 255, value)   # Upper HSV values
        mask = cv2.inRange(hsv_image, lower_range, upper_range)
        ret, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite("1.jpg",thresh)
        return  
    def extractInformation(self, image_path, value):
        self.trainModel()
        self.preprocess_image(image_path, value)
        studentID = image_path.split('.')[0]
        im = cv2.imread('1.jpg')
        im3 = cv2.imread(image_path)
        im = cv2.resize(im, (1149, 726))
        im3 = cv2.resize(im3, (1149, 726))
        out = np.zeros(im.shape, np.uint8)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) > 100  and cv2.contourArea(cnt) < 480:
                [x, y, w, h] = cv2.boundingRect(cnt)
                if h > 19 and h<30 and w >12:
                    roi = thresh[y:y + h, x:x + w]
                    roismall = cv2.resize(roi, (10, 10))
                    roismall = roismall.reshape((1, 100))
                    roismall = np.float32(roismall)
                    retval, results, neigh_resp, dists = self.model.findNearest(roismall, k=1)
                    if(int((results[0][0]))>47 and int((results[0][0]))<59) and y> 200:
                        if (y>500 and x<450) == False:
                            cv2.rectangle(im3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # extract StudentID
                        if y>=450 and y<475 and x<450:
                        # int(chr(int((results[0][0]))))
                            string = str(chr(int(results[0][0])))
                            cv2.putText(out, string, (x, y + h), 0, 1, (0, 255, 0))
        cv2.imwrite(f'result_{studentID}.jpg', im3)
        cv2.imwrite(f'result_studentID_{studentID}.jpg', out)
        return
# test
samples = np.loadtxt('generalsamples.data', np.float32)
responses = np.loadtxt('generalresponses.data', np.float32)
responses = responses.reshape((responses.size, 1))
test = CheckWithKNN(samples, responses)
test.extractInformation('52100796.jpg', 102)
test.extractInformation('52100104.jpg', 89)
test.extractInformation('52100832.jpg', 82)
test.extractInformation('52100018.jpg', 92)
test.extractInformation('52100570.jpg', 92)
