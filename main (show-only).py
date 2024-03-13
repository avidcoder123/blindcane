from ultralytics import YOLO
from PIL import Image
import cv2
import matplotlib as matplot
import matplotlib.pyplot as plt
import time
import pyttsx3

engine = pyttsx3.init()

import sys

model = YOLO("yolov8n.pt")

print("Initializing camera...")

cap = cv2.VideoCapture(0)

#Minimum confidence for detection
conf_threshold = 1/3
#Recently reported class names
reported=dict()

try:
    while cap.isOpened():
        for _ in range(10)
            success, frame = cap.read()
            
        if success:
            cv2.imwrite("./frame.jpeg", frame)
            results = model.track(frame, persist=True, verbose=False)
            classes = [(results[0].names[int(x)], float(y)) for x,y in zip(results[0].boxes.cls, results[0].boxes.conf)]
            classes = list(filter(lambda x: x[1] > conf_threshold, classes))
            classNames = set([x[0] for x in classes])
            toReport = []
            reportFrequency = 1

            for name in classNames:
                if time.time()-reported.get(name, 0) > reportFrequency:
                    toReport.append(name)
                    reported[name] = time.time()

            print(toReport)
            [engine.say(name) for name in toReport]
            engine.runAndWait()
except Exception as err:
    print(err)
    cap.release()
    sys.exit()

plt.close()
cap.release()
