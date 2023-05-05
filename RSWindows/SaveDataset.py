import cv2
import numpy as np
from pathlib import Path
import os

from Detector import Detector

detector = Detector()

datasetDirectory = Path("Dataset")
trainingVideo = os.path.join(datasetDirectory, "TrainingVideo.mp4")

vidcap = cv2.VideoCapture(str(trainingVideo))
success,image = vidcap.read()
count = 0

while success:
    print("At Frame: ", count)
    _, _, unrecognizedDigits = detector.Detect(image, True)

    for img in unrecognizedDigits:
        if np.sum(img) != 0:
            cv2.imshow("Check out this image!", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            print("Enter label:")
            label = int(input())
            
            detector.AddNewData(img, label)

        detector.SaveDataset()

    success,image = vidcap.read()
    count += 1