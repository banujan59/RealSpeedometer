import cv2
import numpy as np
import pickle
from pathlib import Path
import os
import math

class Detector:
    def __init__(self):
        self._HUD_WIDTH = 135
        self._HUD_HEIGHT = 135
        self._HUD_Y = 327

        self._HUD_SPEED_X = 31
        self._HUD_SPEED_Y = 88
        self._HUD_SPEED_WIDTH = 72
        self._HUD_SPEED_HEIGHT = 46

        self._HUD_CENTER_X = 27
        self._HUD_CENTER_Y = 50
        self._HUD_CENTER_WIDTH = 81
        self._HUD_CENTER_HEIGHT = 32

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

        # crop img to get ROI (the speedometer HUD)
        if self._speedometerPositionX is None:
            self.__GetSpeedometerHUDXPositionFromAspectRatio(img)
            
        x = self._speedometerPositionX
        y = self._HUD_Y
        w = self._HUD_WIDTH
        h = self._HUD_HEIGHT
        img = img[y:y+h,x:x+w]

        # Crop & detect the speed
        x = self._HUD_SPEED_X
        y = self._HUD_SPEED_Y
        w = self._HUD_SPEED_WIDTH
        h = self._HUD_SPEED_HEIGHT
        speedImg = img[y:y+h,x:x+w]
        speedImgDetected, prediction, unrecognizedDigits = self.__DetectSpeed(speedImg)
        detectedSpeed = 0

        if len(unrecognizedDigits) == 0 and prediction != "":
          detectedSpeed = int(prediction)

        # Detect RPM
        # Mask the speed
        speedImg.fill(0) # Since it is passed by reference, it will reflect on img too

        # Mask center part of HUD
        x = self._HUD_CENTER_X
        y = self._HUD_CENTER_Y
        w = self._HUD_CENTER_WIDTH
        h = self._HUD_CENTER_HEIGHT
        centerHUD = img[y:y+h,x:x+w]
        centerHUD.fill(0)

        detectedRPM = self.__DetectRPM(img)

        # return results
        return detectedSpeed, detectedRPM

    # TODO update this function
    def __GetSpeedometerHUDXPositionFromAspectRatio(self, img):
        """
        This function will set self._speedometerPositionX acording to the current monitor aspect ratio

        :param img: This should be reszied screenshot buffer (not cropped). The function will use the objet's shape attribute to determine 
        the aspect ratio of the monitor.
        
        """
        dividFactor = img.shape[0] / 9
        widthAspect = round(img.shape[1] / dividFactor)

        if widthAspect == 16: # 16:9 ratio
            self._speedometerPositionX = 700
        elif widthAspect == 21: # 21:9 ratio
            self._speedometerPositionX = 1005
        elif widthAspect == 32: # 32:9 ratio
            self._speedometerPositionX = 1551
        else:
            print("ERROR: The aspect ratio of your monitor is not suported by this application.")
            self._speedometerPositionX = 0


    def __DetectSpeed(self, img):
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


    def __CalculateDistance(self, coordinate1, coordinate2):
      # TODO doc 
      x1,y1 = coordinate1
      x2,y2 = coordinate2
      
      deltaX = pow(x2-x1,2)
      deltaY = pow(y2-y1,2)
      return math.sqrt(deltaX + deltaY)

    def __UniteContours(self, a,b):
      x = min(a[0], b[0])
      y = min(a[1], b[1])
      w = max(a[0]+a[2], b[0]+b[2]) - x
      h = max(a[1]+a[3], b[1]+b[3]) - y
      return (x, y, w, h)

    def __IsValidDigitOnSpeedometerCircle(self, detectedDigitsCoordinates, speedBox):
      """
      Function to position the numbers on the speedometer circle.

      :param detectedDigitsCoordinates: The detected coordinates
      :param speedBox: The bounding box of the current digit to detect
      """
      distanceThresholdBetweenDigits = 30
      
      validDigit = True
      currentSpeedCoordiante = (speedBox[0], speedBox[1])

      for detected in detectedDigitsCoordinates: # TODO is there a better data structure to avoid this ?
        if self.__CalculateDistance(detected, currentSpeedCoordiante) <= distanceThresholdBetweenDigits:
          return False

      return validDigit

    def __ReorderSpeedometerDigits(self, detectedDigitsCoordinates, centerX):
      """
      Function order a list of points representing the digits on the speedometer
      circle. The order will be in increasing number.

      :param detectedDigitsCoordinates: The unordered list of digit's coordinates
      :param centerX: The center of the speedometer circle on the X axis. Will
      help in reordering since the left side is increasing from bottom to top and 
      the right side is increasing from top ot bottom.
      """
      distanceThresholdBetweenDigits = 50 # TODO refactor this between to global variable
      yDistanceThreshold = 30

      # Assuming the list is already sorted by y descending
      listToReturn = []
      
      for coordinate in detectedDigitsCoordinates:
        coordinateX = coordinate[0]
        coordinateY = coordinate[1]
        coordinateInLeftSide = coordinateX < centerX

        inserted = False
        for i,val in enumerate(listToReturn):
          listX = val[0]
          listY = val[1]
          listCoordInLeftSide = listX < centerX

          distance = self.__CalculateDistance(val, coordinate)

          if distance < distanceThresholdBetweenDigits:
            if coordinateInLeftSide != listCoordInLeftSide: # Not on the same side
              yDistance = self.__CalculateDistance((0,listY), (0, coordinateY))
              if yDistance < yDistanceThreshold:
                # we only end up here if we are at the top of the speedometer circle
                if coordinateX < listX:
                  listToReturn.insert(i, coordinate)
                else:
                  listToReturn.insert(i+1, coordinate)
                inserted = True
                break

            # This part is if the coordinates are in the same side of the center:
            else:
              if(coordinateX < centerX):
                # Left side of speedometer. Numbers increase from bottom to top
                if(coordinateY > listY):
                  listToReturn.insert(i, coordinate)
                else:
                  listToReturn.insert(i+1, coordinate)
              else:
                # Right side of speedometer. Numbers increase from top to bottom
                if(coordinateY > listY):
                  listToReturn.insert(i+1, coordinate)
                else:
                  listToReturn.insert(i, coordinate)

              inserted = True
              break
          
        if not inserted:
          listToReturn.append(coordinate)

      return listToReturn

    def __BuildSpeedometer(self, withoutRPMNeedle):
      detectedDigitsCoordinates = []

      # Detect letters position 
      speedContours, hierarchy = cv2.findContours(image=withoutRPMNeedle, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
      withoutRPMNeedle = cv2.cvtColor(withoutRPMNeedle, cv2.COLOR_GRAY2RGB)

      # If there are too many points or not enough, it is not a valid speedometer at target position
      if len(speedContours) > 20 or len(speedContours) < 7:
        return []

      for speedContour in speedContours:
        speedBox = cv2.boundingRect(speedContour)

        if self.__IsValidDigitOnSpeedometerCircle(detectedDigitsCoordinates, speedBox):
          currentXY = (speedBox[0],speedBox[1])
          withoutRPMNeedle = cv2.rectangle(withoutRPMNeedle, currentXY, (speedBox[0]+speedBox[2],speedBox[1]+speedBox[3]), (0,255,0), 1)
          detectedDigitsCoordinates.append(currentXY)

      # TODO recontruct missing digits (in the redline zone...)

      centerX = int(withoutRPMNeedle.shape[1] / 2)
      return self.__ReorderSpeedometerDigits(detectedDigitsCoordinates, centerX)

    def __EstimateCurrentRPM(self, speedometerDigits, rpmBox):
      distanceThresholdBetweenDigits = 50 # TODO refactor this between to global variable

      rpmBoxX = rpmBox[0]
      rpmBoxY = rpmBox[1]
      rpmNeedleCoord = (rpmBoxX, rpmBoxY)
      # TODO adjust rpmNeedleCoord
      # If on the left side, the middle of the left of rpmBox should be taken
      # If on the right side, the middle of the right side of rpmBox should be taken

      lowerBond = ()
      upperBond = ()
      distanceFromLowerBound = 0

      thousandsCounter = 0
      for coordinate in speedometerDigits:
        distance = self.__CalculateDistance(rpmNeedleCoord, coordinate)

        if distance < distanceThresholdBetweenDigits:
          if len(lowerBond) == 0:
            lowerBond = (thousandsCounter, coordinate)
            distanceFromLowerBound = distance

          elif len(upperBond) == 0:
            upperBond = (thousandsCounter, coordinate)
            break

        thousandsCounter = thousandsCounter + 1000

      if len(lowerBond) < 2 or len(upperBond) < 2:
        return 0

      distanceBetweenBounds = self.__CalculateDistance(lowerBond[1], upperBond[1])
      distancePercent = distanceFromLowerBound / distanceBetweenBounds * 100.0

      rpm = distancePercent * 10 + lowerBond[0] # (Equal to distancePercent * 1000 / 100 + lowerBound[0])

      return round(rpm)

    def __DetectRPM(self, img):
      ret, tresh = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)
      kernel = np.ones((3,3),np.uint8)
      rpmIndicator = cv2.erode(tresh,kernel,iterations = 1)
      rpmIndicator = cv2.dilate(rpmIndicator,kernel,iterations = 1) # TODO adjust this. Needle is too thick

      # position of needle
      contours, hierarchy = cv2.findContours(image=rpmIndicator, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)

      if len(contours) > 3: # Not a valid speedometer if we couldn't detect the needle properly
        return 0

      # sometimes, because of the erosion performed previously, the needle might split up
      if len(contours) > 1:
        box1 = cv2.boundingRect(contours[0])
        box2 = cv2.boundingRect(contours[1])
        rpmBox = self.__UniteContours(box1, box2)
      elif len(contours) == 1:
        rpmBox = cv2.boundingRect(contours[0])
      else:
        return 0

      x,y,w,h = rpmBox

      # Remove RPM needle
      withoutRPMNeedle = tresh # TODO
      rpmNeedleInThresholdedImg = withoutRPMNeedle[y:y+h+1,x:x+w+1]
      rpmNeedleInThresholdedImg.fill(0)

      # Build the speedometer
      detectedDigitsCoordinates = self.__BuildSpeedometer(withoutRPMNeedle) # TODO add something to not rebuild the speedometer everytime
      if detectedDigitsCoordinates == []:
        return 0

      return self.__EstimateCurrentRPM(detectedDigitsCoordinates, rpmBox)