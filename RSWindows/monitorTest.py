import argparse
import cv2
import mss
import numpy as np

if __name__ == "__main__":
    print("Welcome! This application will show you a preview of the monitor to capture based on the input index.")
    ap = argparse.ArgumentParser()
    ap.add_argument("-monitor", "--monitor", required=True,
                    help="The index of the monitor where the game will run. Must be an integer. Start with 0.")

    args = vars(ap.parse_args())

    try:
        monitorID = int(args['monitor'])
    except:
        monitorID = -1


    with mss.mss() as sct:
        if monitorID < 0 or monitorID > len(sct.monitors) - 1:
            print("Invalid index. With your current setup, you must choose an index from 0 to", (len(sct.monitors) - 1))
            exit(0)

        monitor = sct.monitors[monitorID] 
        img = np.array(sct.grab(monitor))

        print("Here is a preview of monitor with index", monitorID)
        print("If this is not the correct monitor to capture, re-run this application with a different index.")
        
        dividFactor = 3
        width = int(img.shape[1] / dividFactor)
        height = int(img.shape[0] / dividFactor)
        img = cv2.resize(img, (width, height))
        cv2.imshow("Is this the correct monitor?", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
