from copy import deepcopy
from ultralytics import YOLO
from PIL import Image
import cv2
import matplotlib as matplot
import matplotlib.pyplot as plt
import time
import pyttsx3
import math
import torch
from objSizes import meterSizes
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.data.augment import LetterBox
#import pygame
#import pygame.camera
from picamera2 import Picamera2
engine = pyttsx3.init()
engine.say("Starting program")
engine.runAndWait()
#def tracefunc(frame, event, arg, indent=[0]):
#      if event == "call":
#          indent[0] += 2
#          print("-" * indent[0] + "> call function", frame.f_code.co_name)
#      elif event == "return":
#          print("<" + "-" * indent[0], "exit function", frame.f_code.co_name)
#          indent[0] -= 2
#      return tracefunc

import sys
#sys.setprofile(tracefunc)
# cls = False

# if cls:
#     model = YOLO('yolov8n-cls.pt')
# else:
#     model = YOLO('yolov8n.pt')



# #results = model.train(data = "coco128.yaml", epochs = 5)

# #results = model.val()

# result = model.predict("Collage.png", save_conf=True, verbose=False)[0]
# if cls:
#     classname = [result.names[x] for x in result.probs.top5]
# else:
#     classname = [result.names[int(x)] for x in result.boxes.cls]

# print(classname)

def plot(
    result_obj, distanceMap
):
    """
    Plots the detection results on an input RGB image. Accepts a numpy array (cv2) or a PIL Image.
    Args:
        conf (bool): Whether to plot the detection confidence score.
        line_width (float, optional): The line width of the bounding boxes. If None, it is scaled to the image size.
        font_size (float, optional): The font size of the text. If None, it is scaled to the image size.
        font (str): The font to use for the text.
        pil (bool): Whether to return the image as a PIL Image.
        img (numpy.ndarray): Plot to another image. if not, plot to original image.
        im_gpu (torch.Tensor): Normalized image in gpu with shape (1, 3, 640, 640), for faster mask plotting.
        kpt_radius (int, optional): Radius of the drawn keypoints. Default is 5.
        kpt_line (bool): Whether to draw lines connecting keypoints.
        labels (bool): Whether to plot the label of bounding boxes.
        boxes (bool): Whether to plot the bounding boxes.
        masks (bool): Whether to plot the masks.
        probs (bool): Whether to plot classification probability
        show (bool): Whether to display the annotated image directly.
        save (bool): Whether to save the annotated image to `filename`.
        filename (str): Filename to save image to if save is True.
    Returns:
        (numpy.ndarray): A numpy array of the annotated image.
    Example:
        ```python
        from PIL import Image
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        results = model('bus.jpg')  # results list
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            im.show()  # show image
            im.save('results.jpg')  # save image
        ```
    """
    conf=True
    line_width=None
    font_size=None
    font="Arial.ttf"
    pil=False
    img=None
    im_gpu=None
    kpt_radius=5
    kpt_line=True
    labels=True
    boxes=True
    masks=True
    probs=True
    show=False
    save=False
    filename=None
    if img is None and isinstance(result_obj.orig_img, torch.Tensor):
        img = (result_obj.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()
    names = result_obj.names
    is_obb = True if hasattr(result_obj, 'obb') and result_obj.obb else False
    pred_boxes, show_boxes = result_obj.obb if is_obb else result_obj.boxes, boxes
    pred_masks, show_masks = result_obj.masks, masks
    pred_probs, show_probs = result_obj.probs, probs
    annotator = Annotator(
        deepcopy(result_obj.orig_img if img is None else img),
        line_width,
        font_size,
        font,
        pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
        example=names,
    )
    # Plot Segment results
    if pred_masks and show_masks:
        if im_gpu is None:
            img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
            im_gpu = (
                torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device)
                .permute(2, 0, 1)
                .flip(0)
                .contiguous()
                / 255
            )
        idx = pred_boxes.cls if pred_boxes else range(len(pred_masks))
        annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)
    # Plot Detect results
    if pred_boxes is not None and show_boxes:
        for d in reversed(pred_boxes):
            c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
            name = ("" if id is None else f"id:{id} ") + names[c]
            label = (f"{name} {distanceMap[name]} steps" if conf else name) if labels else None
            box = d.xyxyxyxy.reshape(-1, 4, 2).squeeze() if is_obb else d.xyxy.squeeze()
            annotator.box_label(box, label, color=colors(c, True))
    # Plot Classify results
    if pred_probs is not None and show_probs:
        text = ",\n".join(f"{names[j] if names else j} {pred_probs.data[j]:.2f}" for j in pred_probs.top5)
        x = round(result_obj.orig_shape[0] * 0.03)
        annotator.text([x, x], text, txt_color=(255, 255, 255))  # TODO: allow setting colors
    # Plot Pose results
    if result_obj.keypoints is not None:
        for k in reversed(result_obj.keypoints.data):
            annotator.kpts(k, result_obj.orig_shape, radius=kpt_radius, kpt_line=kpt_line)
    # Show results
    if show:
        annotator.show(result_obj.path)
    # Save results
    if save:
        annotator.save(filename)
    return annotator.result()

print("Loading YOLO...")
model = YOLO("/home/pi/yolocanefinal/yolov8s.pt")
print("Done!")

print("Initializing camera...")
#pygame.camera.init()
#cameras = pygame.camera.list_cameras() #Camera detected or not
#print(cameras)
#print("Done!")

display = True
#cap = cv2.VideoCapture("./busride.mp4")
#cap = cv2.VideoCapture(0)
#print("Loaded camera")
# if display:
#     matplot.use("TkAgg")
#     fig, ax = plt.subplots()

#Higher for faster framerate
#skip_frames = 10

#Minimum confidence for detection
conf_threshold = 1/3
#Recently reported class names
reported=dict()
print("Starting program")
timetoprocess = 1
timetoread = 0.1
try:
    frame = cv2.imread("./testroom.jpg")

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
    namesToReport = []
    #function to see if two numbers are within 10% of each other
    areClose = lambda x,y: abs(x - y) / x <= 0.1
    #Only report names which are newly encountered
    distanceMap = dict()
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
        distance = max(round(distance * 1.3),1) #Convert meters to steps
        if distance <= 15:
            toReport.append(f'{" ".join(name)} {distance} step{"s" if distance != 1 else ""}')
            namesToReport.append(name)
            distanceMap[name[0]] = distance
            reported[" ".join(name)] = time.time()
    print(toReport)
    # time.sleep(1)
    # break
    # Visualize the results on the frame
    if display:
        print(distanceMap)
        names = list()
        distanceDisplayList = []
        for name, _ in namesToReport:
            distanceDisplayList.append(distanceMap[name])
        #results[0].boxes.conf = torch.tensor([])
        annotated_frame = plot(results[0], distanceMap)
        # Display the annotated frame
        # ax.clear()
        # ax.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        frame_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite("./testroom_annotated.jpg", frame_image)
        # plt.draw()
        # plt.pause(1e-3)
except Exception as err:
    #cap.release()
    raise err

# plt.close()
#cap.release()
