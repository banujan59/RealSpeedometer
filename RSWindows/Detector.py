import cv2
import numpy as np
import pickle
from pathlib import Path
import os
import math
import sympy

class Detector:
    def __init__(self):
        self._HUD_WIDTH = 135
        self._HUD_HEIGHT = 135
        self._HUD_Y = 317

        self._HUD_SPEED_X = 31
        self._HUD_SPEED_Y = 88
        self._HUD_SPEED_WIDTH = 72
        self._HUD_SPEED_HEIGHT = 50

        self._HUD_CENTER_X = 27
        self._HUD_CENTER_Y = 50
        self._HUD_CENTER_WIDTH = 81
        self._HUD_CENTER_HEIGHT = 40

        self._speedometer_centerX = 0
        self._speedometer_centerY = 0
        self._speedometer_radius = 0
        self._speedometer_center_angle = 0.0
        self._speedometer_distance_between_digits = 0

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

    # Function will return prediction and unrecognized digits (if in training mode)
    def Detect(self, img, training=False):
        
        # debugFilename = os.path.join("frameDebug", "frame_" + str(self.__frameDebugCounter) + ".png")
        # cv2.imwrite(str(debugFilename), img)
        # self.__frameDebugCounter = self.__frameDebugCounter + 1
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

        detectedRPM = 0

        if not training:
          detectedRPM = self.__DetectRPM(img)

        # return results
        if training == True:
           return detectedSpeed, detectedRPM, unrecognizedDigits
        else:
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
            self._speedometerPositionX = 1531
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
                            
                            #print("Unknown. LowestSum: ", lowestSum, " Prediction: ", self.labels[lowestSumIndex])

                        unrecognizedDigits.append(digitImg)

                return img, prediction, unrecognizedDigits

        return img, "", []


    def __CalculateDistance(self, coordinate1, coordinate2):
      """
      Function to calculate the euclidean distance between 2 coordinates.
      Euclidean distance formula: d = (x2-x1)^2 + (y2-y1)^2

      :param coordinate1: The x1,y1 coordinate to use in the euclidean distance
      :param coordinate2: The x2,y2 coordinate to use in the euclidean distance
      :return: The distance between points coordinate1 and coordinate2
      """
      x1,y1 = coordinate1
      x2,y2 = coordinate2
      
      deltaX = pow(x2-x1,2)
      deltaY = pow(y2-y1,2)
      return math.sqrt(deltaX + deltaY)

    def __ReorderSpeedometerDigits(self, detectedDigitsCoordinates, centerX):
      """
      Function order a list of points representing the digits on the speedometer
      circle. The order will be in increasing RPM.

      :param detectedDigitsCoordinates: The unordered list of digit's coordinates
      :param centerX: The center of the speedometer circle on the X axis. Will
      help in reordering since the left side is increasing from bottom to top and 
      the right side is increasing from top ot bottom.
      """
      yDistanceThreshold = 30
      distanceTooFar = 35

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

          if distance > distanceTooFar:
            continue

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

          # If the coordinates are in the same side of the center:
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
    
    def __GetRPMNeedle(self, img):
      """
      Function to detect the RPM needle on the speedometer circle.
      :param img: The image of the speedometer. Some pre-processing should be applied to make the image sharper.
      :return rpmNeedleCandidates: The candidates of points that could represent the RPM needle. 
      :return withoutRPMNeedle: The image of the speedometer but without the possible RPM candidates. This will be used later to detect the digits.
      """
      _, withoutRPMNeedle = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
      kernel = np.ones((2,2),np.uint8)
      rpmIndicator = cv2.erode(withoutRPMNeedle,kernel,iterations = 1)
      kernel = np.ones((2,2),np.uint8)
      rpmIndicator = cv2.dilate(rpmIndicator,kernel,iterations = 2)

      # position of needle
      contours, _ = cv2.findContours(image=rpmIndicator, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)

      if len(contours) == 0 or len(contours) > 8: # Not a valid speedometer if we couldn't detect the needle properly
        return [], np.array([])

      # sometimes, because of the erosion performed previously, there might be noise left...
      # Find the biggest contours among all the contours detected
      idealRPMNeedleMaxSize = 22 # Reject all contours above this size
      idealRPMNeedleMinSize = 8 # Reject all contours below this size
      rpmNeedleCandidates = []

      for contour in contours:
        currentRpmBox = cv2.boundingRect(contour)
        x,y,w,h = currentRpmBox
        currentSize = w * h

        if currentSize >= idealRPMNeedleMinSize and currentSize <= idealRPMNeedleMaxSize:
          rpmNeedleCandidates.append(currentRpmBox)
        
        # Remove all contours to be left with only the digits
        rpmNeedleInThresholdedImg = withoutRPMNeedle[y:y+h+1,x:x+w+1]
        rpmNeedleInThresholdedImg.fill(0) # TODO what happens if needle is on a digit used to reconstruct the speedometer?

      kernel = np.ones((1,1),np.uint8)
      withoutRPMNeedle = cv2.erode(withoutRPMNeedle,kernel,iterations = 5)
      kernel = np.ones((2,2),np.uint8)
      withoutRPMNeedle = cv2.dilate(withoutRPMNeedle,kernel,iterations = 2)

      return rpmNeedleCandidates, withoutRPMNeedle

    def __GetSpeedometerDigitPositions(self, withoutRPMNeedle):
      detectedDigitsCoordinates = []

      yOffset = 25
      x = 0
      y = yOffset
      w = 45
      h = 100
      leftSideRPM = withoutRPMNeedle[y:y+h,x:x+w]

      # Detect letters position 
      speedContours, _ = cv2.findContours(image=leftSideRPM, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)

      # If there are too many points or not enough, it is not a valid speedometer at target position
      if len(speedContours) > 15 or len(speedContours) < 3:
        return []

      maxYAccepted = 94 # Contours above this will be rejected
      digitsDimentions = [
        (5,6),  # For digits 0,2,3,4
        (2,5)  # For digit 1
      ]
      
      SIZE_TOLERANCE = 2
      for speedContour in speedContours:
        speedBox = cv2.boundingRect(speedContour)

        # Adjust y coordinate since we cropped before detecting contours
        x,y,w,h = speedBox
        
        if y > maxYAccepted:
          continue

        # Correct the y since we cropped the left part of the speedometer
        y = y+yOffset

        validDigit = False
        for digitsDimention in digitsDimentions:
          if validDigit:
            continue

          expectedWidth = digitsDimention[0]
          expectedHeight = digitsDimention[1]
          widthDifferenceAcceptable = abs(expectedWidth - w) < SIZE_TOLERANCE
          heightDifferenceAcceptable = abs(expectedHeight - h) < SIZE_TOLERANCE

          if widthDifferenceAcceptable and heightDifferenceAcceptable:
            detectedDigitsCoordinates.append( (int(x + w/2), int(y + h/2)) )
            validDigit = True

      centerX = int(withoutRPMNeedle.shape[1] / 2)
      return self.__ReorderSpeedometerDigits(detectedDigitsCoordinates, centerX)

    def __BuildSpeedometer(self, detectedDigitsCoordinates):
      """
      This function will rebuild the speedometer and get the coordinates of all possible RPMs.
      
      :param detectedDigitsCoordinates: An array of the detected contours of the RPMs (their x and y posiitons) from function __GetRPMDigitContours. 
              It is expected that this list is already ordered by increasing RPMs. 
      """
      speedometerDigits = []

      if len(detectedDigitsCoordinates) >= 3:
        # Step 1: Find the equation of the circle to find radius and its center.
        # The equation of a circle is: (x-h)^2 + (y-k)^2 = r^2
        # Therefore, we need at least 3 points solve the equation
        x1 = detectedDigitsCoordinates[0][0]
        y1 = detectedDigitsCoordinates[0][1]

        x2 = detectedDigitsCoordinates[1][0]
        y2 = detectedDigitsCoordinates[1][1]

        x3 = detectedDigitsCoordinates[2][0]
        y3 = detectedDigitsCoordinates[2][1]

        r, h, k = sympy.symbols('r h k')
        results = sympy.solve([sympy.Eq(pow((x1-h),2) + sympy.Pow((y1-k),2), sympy.Pow(r,2)), sympy.Eq(sympy.Pow((x2-h),2) + sympy.Pow((y2-k),2), sympy.Pow(r,2)), sympy.Eq(sympy.Pow((x3-h),2) + sympy.Pow((y3-k),2), sympy.Pow(r,2))], [r, h, k])

        if len(results) < 2:
          return [] # Error

        self._speedometer_centerX = 0
        self._speedometer_centerY = 0
        self._speedometer_radius = 0

        # There should be 2 answers because of the sqrt of radius
        for result in results:
          # Reject the answer with the negative radius.
          if result[0] < 0:
            continue
          r,h,k = result
          radius = float(r)
          centerX = round(float(h))
          centerY = round(float(k))

        # Step 2: Find the center angle of the arc formed by 2 consecutive RPM
        # FInd the euclidient distance between 2 consecutive points
        # Assumption: the array passed to this function is already sorted by RPM
        coordRPM1 = detectedDigitsCoordinates[0]
        coordRPM2 = detectedDigitsCoordinates[1]
        distanceBetweenRPM = self.__CalculateDistance(coordRPM1, coordRPM2)
        self._speedometer_distance_between_digits = distanceBetweenRPM

        # Divide distance by 2 to construct right a angle triangle 
        d = distanceBetweenRPM / 2.0

        # Simple trigonometry to find half of the center angle
        centerAngle = math.asin(d / radius)

        # Multiply by 2 to get center angle
        centerAngle = centerAngle * 2

        # Step 3: Find angle relative to first quadrant
        rpm1X = coordRPM1[0]
        angleFromFirstQuadrant = math.acos((rpm1X - centerX) / radius)

        # Step 4: Populate the RPM list
        # Just keep adding the center angle to the reference angle
        # and use simple trigonometry to find the coordinates of the RPMs
        # Assumption: Every arc formed by RPM have same center angle (RPM are equally spaced)

        # The number of possible RPM in this case is 360 / center angle of arc (in deg) + 1
        centerAngleDeg = centerAngle * 180 / math.pi
        numberOfRPMs = int(360 / centerAngleDeg) + 1

        for i in range(numberOfRPMs):
          if i > 0:
            angleFromFirstQuadrant = angleFromFirstQuadrant + centerAngle

          rpmX = math.cos(angleFromFirstQuadrant) * radius + centerX
          rpmY = math.sin(angleFromFirstQuadrant) * radius + centerY

          rpmX = round(rpmX)
          rpmY = round(rpmY)

          speedometerDigits.append( (rpmX, rpmY) )

        self._speedometer_centerX = centerX
        self._speedometer_centerY = centerY
        self._speedometer_radius = radius
        self._speedometer_center_angle = centerAngle

      return speedometerDigits

    def __EstimateCurrentRPM(self, speedometerDigits, rpmNeedleCandidates):
      if len(rpmNeedleCandidates) == 0:
        return 0

      # Estimate the best needle based on the circle's equation
      # The idea is to insert the x and y of each possible rpm needle into the circle's equation.
      # Then, we will pick the best RPM needle candidate based on how far it is from radius squared 
      # (remember the equation of the circle is (x-h)^2 + (y-k)^2 = r^2)
      # This will tell us how far the candidate is from the border of the circle (we assume the real needle touches the border)
      # TODO this algorithm might have a flaw: What if the "needle" detected is not correct because it is noise that is very close to the circle?
      chosenRPMBox = (0,0,0,0)
      smallestDifference = 6540 # An arbitrary value. 
      radiusSquared = pow(self._speedometer_radius, 2)

      for rpmBox in rpmNeedleCandidates:
        x,y,w,h = rpmBox
        rpmBoxX = int(x + w/2)
        rpmBoxY = int(y + h/2)

        # Insert into circle's equation
        result = pow(rpmBoxX - self._speedometer_centerX,2) + pow(rpmBoxY - self._speedometer_centerY,2)
        difference = abs(radiusSquared - result)

        if difference < smallestDifference:
          smallestDifference = difference
          chosenRPMBox = rpmBox

      # Take the center of the RPM needle for a better approximation
      x,y,w,h = chosenRPMBox
      rpmBoxX = int(x + w/2)
      rpmBoxY = int(y + h/2)
      rpmNeedleCoord = (rpmBoxX, rpmBoxY)

      # Construct line from middle of rpm needle to the center of the circle
      # Equation is: y = ax + b
      a = (rpmBoxY - self._speedometer_centerY) / (rpmBoxX - self._speedometer_centerX)
      b = rpmBoxY - a * rpmBoxX
      
      # Solve for the 2 points where the line touches in the speedometer circle
      # The equation of a circle is: (x-h)^2 + (y-k)^2 = r^2. 
      # If we substitute the y in the circle equation by the line equation, we get:
      # (x-h)^2 + ((a*x+b)-k)^2 = r^2. 
      # Now we solve for x to get the two points on the circle.
      possibleXs = list()
      x = sympy.symbols('x')
      results = sympy.solve([sympy.Eq(sympy.Pow((x - self._speedometer_centerX),2) + sympy.Pow(((a * x + b) - self._speedometer_centerY),2), sympy.Pow(self._speedometer_radius,2))], [x])

      if len(results) != 2: # Should always return 2 results. If not, the line/circle equation is incorrect.
        return 0

      possibleXs.append(results[0][0])
      possibleXs.append(results[1][0])

      # Pick the point that is closest to the needle point
      chosenPointOnCircle = (0,0)
      closestDistance = -1

      for x in possibleXs:
        y = a*x + b
        distance = self.__CalculateDistance(rpmNeedleCoord, (x,y))

        if closestDistance == -1 or distance < closestDistance:
          closestDistance = distance
          chosenPointOnCircle = (x,y)

      # Find between which RPM is the point located
      thousandsCounter = 0
      anglePercent = 0.0

      for coordinate in speedometerDigits:
        # Calculate distance between point and current RPM
        distance = self.__CalculateDistance(chosenPointOnCircle, coordinate)

        # If distance is greater than distance between digit, the point doesn't belong to the current RPM
        if distance < self._speedometer_distance_between_digits:
          # Calculate angle from point to RPM start
          d = distance / 2.0
          angleFromRPM = math.asin(d / self._speedometer_radius) * 2
          
          # Calculate the percent of the angle
          anglePercent = angleFromRPM / self._speedometer_center_angle * 100.0
          break

        thousandsCounter = thousandsCounter + 1000

      rpm = anglePercent * 10 + thousandsCounter # (Equal to anglePercent * 1000 / 100 + thousandsCounter)
      return round(rpm)

    def __DetectRPM(self, img):
      edgeKernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
      img = cv2.filter2D(img, -1, edgeKernel)

      rpmNeedleCandidates, withoutRPMNeedle = self.__GetRPMNeedle(img)
      if len(rpmNeedleCandidates) == 0: # A proper sized needle could not be detected.
        return 0

      # Build the speedometer
      detectedDigitsCoordinates = self.__GetSpeedometerDigitPositions(withoutRPMNeedle) # TODO add something to not rebuild the speedometer everytime
      if detectedDigitsCoordinates == []:
        return 0

      speedometerDigits = self.__BuildSpeedometer(detectedDigitsCoordinates)
      if speedometerDigits == []:
        return 0

      return self.__EstimateCurrentRPM(speedometerDigits, rpmNeedleCandidates)