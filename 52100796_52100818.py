import cv2
import numpy as np
import os

class CreateTrainingSet:
    def __init__(self, image_path):
        self.image_path = image_path
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
            label_folder = os.path.join(self.image_path, str(label))
            
            # Kiểm tra xem thư mục tồn tại hay không
            if os.path.exists(label_folder):
                # print(f"Processing label {label}: {label_folder}")
                
                # Duyệt qua các tệp trong thư mục
                for filename in os.listdir(label_folder):
                    file_path = os.path.join(label_folder, filename)
                    im = cv2.imread(file_path)

                        # Thực hiện công việc cần thiết với mỗi tệp trong thư mục
                    thresh =self.preprocessing_image(im)
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
                                    self.responses.append(ord(label))
                                    # print(ord(label))
                                    sample = roismall.reshape((1, 100))
                                    self.samples = np.append(self.samples, sample, 0)

                                    # cv2.imshow('Contours', im)
                                    # cv2.waitKey(0)
                                    # cv2.destroyAllWindows()
                                    # responses.append(label)  # Gán nhãn tự động dựa trên số thư mục
                                    # sample = roismall.reshape((1, 100))
                                    # samples = np.append(samples, sample, 0)

                        # Hiển thị hình ảnh với các đường viền
                        
                
            else:
                print(f"Label {label} folder does not exist.")
        np.savetxt('generalsamples.data', self.samples)
        np.savetxt('generalresponses.data', self.responses)
        return "Ok"
    
# training complete so I comment this code =))
# image_path = f'training_data'
# train = CreateTrainingSet(image_path)
# train.training()

class CheckWithKNN:
    def __init__(self, samples, responses):
        self.samples = samples
        self.responses = responses
        self.model = cv2.ml.KNearest_create()
    def trainModel(self):
        self.model.train(self.samples, cv2.ml.ROW_SAMPLE, self.responses)
    def preprocess_image(self, image_path):
        input_image = cv2.imread(image_path)
        kernel = np.ones((5,5),np.uint8)
        hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
        lower_orange = (0, 0, 0)  # Lower HSV values for star color
        upper_orage = (179, 255,92)   # Upper HSV values for star color

        mask = cv2.inRange(hsv_image, lower_orange, upper_orage)
        ret, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)

        # mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.imwrite("1.jpg",thresh)
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
    def extractInformation(self, image_path):
        self.trainModel()
        imSize = cv2.imread('52100796.jpg')
        self.preprocess_image(image_path)
        im = cv2.imread('1.jpg')
        im3 = cv2.imread(image_path)
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
                    retval, results, neigh_resp, dists = self.model.findNearest(roismall, k=1)
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
samples = np.loadtxt('generalsamples.data', np.float32)
responses = np.loadtxt('generalresponses.data', np.float32)
responses = responses.reshape((responses.size, 1))
test = CheckWithKNN(samples, responses)
test.extractInformation('52100832.jpg')