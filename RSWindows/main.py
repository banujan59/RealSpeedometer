import argparse
import mss
import time
import numpy as np
from Detector import Detector
from ClusterCommunicator import ClusterCommunicator
from Car import Car

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-ip", "--ip", required=True,
                    help="The IP address of the embedded device that will control the gauge cluster.")
    ap.add_argument("-p", "--port", required=True,
                    help="The port of the TCP server running on the embedded device.")
    ap.add_argument("-monitor", "--monitor", required=True,
                    help="The index of the monitor where the game will run. Use monitorTest.py to see if your index points to the correct monitor.")
    ap.add_argument("-disableEmbedded", "--disableEmbedded", required=False, action=argparse.BooleanOptionalAction,
                    help="[optional] If set, the data will not be sent to the embedded device.")
    ap.set_defaults(disableEmbedded=False)

    args = vars(ap.parse_args())
    
    try:
        monitorID = int(args['monitor'])
    except:
        print("Invalid monitor ID: ", args['monitor'])
        print("Use monitorTest.py to test for your correct monitor ID.")
        exit(-1)

    sendToEmbedded = args['disableEmbedded']
    clusterCommunicator = ClusterCommunicator(args["ip"], args["port"], args['disableEmbedded'])

    car = Car()

    detector = Detector()
    detector.LoadDataset()

    detectionTimeSum = 0.0
    nbDetection = 0

    with mss.mss() as sct:
        monitor = sct.monitors[monitorID]
        
        while True:
            img = np.array(sct.grab(monitor))
            
            start = time.time() * 1000000
            detectedSpeed, detectedRPM = detector.Detect(img)
            end = time.time() * 1000000
            duration = end - start
            nbDetection = nbDetection + 1
            detectionTimeSum = detectionTimeSum + duration

            print(f"{detectedSpeed} km/h {detectedRPM} RPM. Detection time is {duration} us, \taverage is {detectionTimeSum / nbDetection} us")
            car.SetData(detectedSpeed, detectedRPM, 0, 0)
            clusterCommunicator.SendData(car.encodeData())

            time.sleep(0.1)
    
