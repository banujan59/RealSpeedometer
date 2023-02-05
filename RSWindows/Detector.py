import cv2
import numpy as np
import pickle
from pathlib import Path
import os

class Detector:
    def __init__(self):
        self.DATASET_WIDTH = 90
        self.DATASET_HEIGHT = 50

        self.datas = []
        self.labels = []

        datasetDirectory = Path("Dataset")

        if not os.path.exists(datasetDirectory):
            os.mkdir(datasetDirectory)

        self._dataFile = os.path.join(datasetDirectory, "datas.pkl")
        self._labelFile = os.path.join(datasetDirectory, "labels.pkl")

        self._speedometerPositionX = None # This will be set during the first call of the Detect function

    def LoadDataset(self):
        with open(self._dataFile, 'rb') as file:
            self.datas = pickle.load(file)

        with open(self._labelFile, 'rb') as file:
            self.labels = pickle.load(file)

        self.datas = np.array(self.datas)
        self.labels = np.array(self.labels)
    
        print(self.datas.shape)
        print(self.labels.shape)

    def SaveDataset(self):
        # Open a file and use dump()
        with open(self._dataFile, 'wb') as file:
            pickle.dump(self.datas, file)
        
        with open(self._labelFile, 'wb') as file:
            pickle.dump(self.labels, file)

    def AddNewData(self, img, label):
        self.datas.append(img)
        self.labels.append(label)

    # Function will return prediction and unrecognized digits
    def Detect(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h = 480
        w = int(img.shape[1] / (img.shape[0] / h))
        img = cv2.resize(img, (w, h))

        # crop img to get ROI (the speedometer)
        if self._speedometerPositionX is None:
            self.__GetSpeedometerHUDXPositionFromAspectRatio(img)
            
        x = self._speedometerPositionX
        y = 413
        img = img[y:y+self.DATASET_HEIGHT,x:x+self.DATASET_WIDTH]
        ret, img = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)

        if np.sum(img) != 0:
            contours, hierarchy = cv2.findContours(image=img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)

            digits = []

            for contour in contours:
                box = cv2.boundingRect(contour)
                x,y,w,h = box

                if h <= 30:
                    continue

                similarDetectFound = False
                for x_other, _ in digits:
                    if abs(x_other - x) < 10:
                        similarDetectFound = True
                        break
                if similarDetectFound:
                    continue

                crop = img[y:y+h, x:x+w]
                digits.append([x, crop])

            if len(digits) <= 3:
                digits.sort(key=lambda x: x[0])

                unrecognizedDigits = []
                prediction = ""

                for _, digitImg in digits:
                    # cv2.imshow("Test", digitImg)
                    # cv2.waitKey(0)

                    digitImg = cv2.resize(digitImg, (25, 45), cv2.INTER_NEAREST)

                    found = False
                    lowestSum = 999999
                    lowestSumIndex = -1

                    for i, val in enumerate(self.datas):
                        differenceImg = val - digitImg
                        kernel = np.ones((5,5),np.uint8)
                        differenceImg = cv2.erode(differenceImg,kernel,iterations = 1)

                        sum = np.sum(differenceImg)

                        if lowestSum > sum:
                            lowestSum = sum
                            lowestSumIndex = i
                        
                        if sum <= 10:
                            found = True
                            prediction = prediction + str(self.labels[i])
                            break
                    
                    if not found:
                        if lowestSumIndex != -1:
                            if int(self.labels[lowestSumIndex]) == 1 and lowestSum <= 100: #  sometimes the number 1 is diffult to detect using normal threashold... 
                                prediction = prediction + str(self.labels[lowestSumIndex])
                                continue
                            
                            print("Unknown. LowestSum: ", lowestSum, " Prediction: ", self.labels[lowestSumIndex])

                        unrecognizedDigits.append(digitImg)

                return img, prediction, unrecognizedDigits

        return img, "", []

    def __GetSpeedometerHUDXPositionFromAspectRatio(self, img):
        """
        This function will set self._speedometerPositionX acording to the current monitor aspect ratio

        :param img: This should be reszied screenshot buffer (not cropped). The function will use the objet's shape attribute to determine 
        the aspect ratio of the monitor.
        
        """
        dividFactor = img.shape[0] / 9
        widthAspect = int(img.shape[1] / dividFactor)

        if widthAspect == 16: # 16:9 ratio
            self._speedometerPositionX = 720
        elif widthAspect == 21: # 21:9 ratio
            self._speedometerPositionX = 1005
        elif widthAspect == 32: # 32:9 ratio
            self._speedometerPositionX = 1551
        else:
            print("ERROR: The aspect ratio of your monitor is not suported by this application.")
            self._speedometerPositionX = 0

