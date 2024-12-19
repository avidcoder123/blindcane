from ultralytics import YOLO
from PIL import Image
import cv2
import matplotlib as matplot
import matplotlib.pyplot as plt
import time
import pyttsx3
import math
from objSizes import meterSizes

from picamera2 import Picamera2
engine = pyttsx3.init()
engine.say("Starting program")
engine.runAndWait()


import sys


print("Loading YOLO...")
model = YOLO("/home/pi/yolocanefinal/yolov8s.pt")
print("Done!")

print("Initializing camera...")

#True to create matplotlib diagrams with the bounding boxes. For debugging only.
display = False

camera = Picamera2()
camera.resolution = (640,480)
camera.start()
if display:
    matplot.use("TkAgg")
    fig, ax = plt.subplots()

#Minimum confidence for detection
conf_threshold = 1/3
#Recently reported class names
reported=dict()
print("Starting program")
timetoprocess = 1
timetoread = 0.1
try:
    while True:
        frame = camera.capture_array("main")
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        if True:
            results = model.predict(frame)
            #Get if object is right left or center
            def getDirection(pixval):
                if pixval <= 0.2:
                    return "to the left"
                elif pixval <= 0.4:
                    return "slight left"
                elif pixval <= 0.6:
                    return "straight ahead"
                elif pixval <= 0.8:
                    return "slight right"
                else:
                    return "to the right"
                
            classNames = results[0].boxes.cls
            probabilities = results[0].boxes.conf
            normalizedPositions = results[0].boxes.xyxyn

            #A list of the objects' width then height
            objSizeImg = list(map(lambda x: (x[2].item(), x[3].item()), results[0].boxes.xywhn))

            #Get class names and probabilities and x-centers
            classes = [(results[0].names[int(x)], float(y), getDirection(((z[0] + z[2])/2).item())) for x,y,z in zip(classNames, probabilities, normalizedPositions)]
            #Filter out using confidence threshold
            classes = list(filter(lambda x: x[1] > conf_threshold, classes))
            #Extract only class names
            classNames = set([(x[0], x[2]) for x in classes])
            #List of names to report via TTS
            toReport = []

            #How long the software should wait before re-reporting an object, in seconds.
            #This way, if an object is continously detected, the text-to-speech will not spam.
            reportFrequency = 1

            #function to see if two numbers are within 10% of each other
            areClose = lambda x,y: abs(x - y) / x <= 0.1

            #Only report names which are newly encountered
            for idx, name in enumerate(classNames):
                distance = 0
                if meterSizes.get(name[0]):
                    #Pixel dimensions (width, height)
                    dimensions = objSizeImg[idx]

                    #Dimensions of real object, meters (width, height)
                    meterDimensions = meterSizes[name[0]]

                    #Function to get width:height ratio of tuple
                    proportion = lambda x: x[0]/x[1]

                    #If the object seems wider than usual (greater width:height ratio), judge based on width
                    if proportion(dimensions) >= proportion(meterDimensions):
                        #Make use of x-direction FOV
                        fov = 54
                        axis = 0 #Use width
                    else:
                        #Use y-direction FOV
                        fov = 41
                        axis = 1 #Use height
                    arcSize = dimensions[axis] * fov
                    distance = meterDimensions[axis]/(2 * math.tan(math.radians(arcSize)/2)) #Gives distance in meters


                if time.time()-reported.get(name, 0) > reportFrequency:
                    distance = max(round(distance * 1.3),1) #Convert meters to steps
                    if distance <= 15:
                        toReport.append(f'{" ".join(name)} {distance} step{"s" if distance != 1 else ""}')
                        reported[" ".join(name)] = time.time()

            print(toReport)
            [engine.say(name) for name in toReport]
            engine.runAndWait()

            # Visualize the results on the frame
            if display:
                annotated_frame = results[0].plot()

                # Display the annotated frame
                ax.clear()
                ax.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                plt.draw()
                plt.pause(1e-3)
except Exception as err:
    raise err

plt.close()