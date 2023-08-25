import random
import time
import sys
import pandas as pd
import datetime
import torch
import cv2
import os
import numpy as np
from ultralytics import YOLO
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from bson.json_util import dumps


def video_detection(videosource,database,new_job_id):
    class_list = ["Hole", "Stain"]
    writer = None

    # Generate random colors for class list
    detection_colors = []
    for i in range(len(class_list)):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        detection_colors.append((b, g, r))
    bbs = []
    confs = []
    class_names = []
    times = []
    hyt=[]
    wid=[]
    frame_counts = []

    total_frames = 0
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    f = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load a pretrained YOLOv8n model
    model = YOLO("weights/sh_new.pt", "v8")
    cap = cv2.VideoCapture(videosource)
    #width = int(cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640))
    #height = int(cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480))
    #fps = cap.set(cv2.CAP_PROP_FPS, 10)

    if not cap.isOpened():
      print("Cannot open camera")
      exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        #frames = frame.copy()
        if frame is None:
            print("End of stream")
            break

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Predict on image/video
        detect_params = model.predict(source=[frame], conf=0.5,verbose=False)

        # Convert tensor array to numpy
        DP = detect_params[0].cpu().numpy()
        #print(DP)
        total_frames += 1

        if len(DP) != 0:
            
            for i in range(len(detect_params[0])):
                #print(i)
                boxes = detect_params[0].boxes
                box = boxes[i]  # returns one box
                masks = detect_params[0].masks
                clsID = box.cls.cpu().numpy()[0]
                conf = (box.conf.cpu().numpy()[0])*100
                confi = "%.2f" % conf
                bb = box.xyxy.cpu().numpy()[0]
                class_name = class_list[int(clsID)]
               # print(boxes, box, clsID,confi,bb)

                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])), # x, y, width, height as x1 y1 x2 y2
                    (int(bb[2]), int(bb[3])),
                    detection_colors[int(clsID)],
                    3,
                )
        
                # Display class name and confidence
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    frame,
                    class_name + " " + str(confi) + "%",
                    (int(bb[0]), int(bb[1]) - 10),
                    font,
                    1,
                    (255, 255, 255),
                    2,
                )

                #define an array named existing classname n push class name 
                if bb[1] > 310 and bb[1] < 350 and int(bb[0]/10) != f:
                    bbs.append(bb)
                    confs.append(confi)
                    class_names.append(class_name)
                    times.append(current_time)
                    frame_counts.append(total_frames)   
                    x,y,w,h = int(bb[0]), int(bb[1]),int(bb[2]), int(bb[3])        
                    h=abs(y-h)
                    w=abs(x-w)     
                    hyt.append(h)
                    wid.append(w)             
                    area = w*h       
                    f = int(bb[0]/10)

                    df = pd.DataFrame({'Time' : times, 'Frame' : frame_counts, 'Class_name' : class_names, 'Conf' : confs, 'width': wid, 'height': hyt}) 
                    df.to_csv('result.csv', index = False)

                    anomalycol=database["Anomaly"]
                    #save detected image
                    img_name = str(class_name)+str(total_frames)+'.jpg'
                    name =r'C:/laragon/www/det_images/3/'+img_name
                    anomalycol.insert_one({"Time": current_time,"Frame": total_frames, "Type":class_name, "Accuracy":confi, "img":img_name, 
                                           "JobId": new_job_id, "Sensor": "SensorA","width": w, "height": h, "Area": area})
                
                    print ('Creating...' + name)
                    cv2.imwrite(name, frame)
        # Initialize the video writer object
        if writer is None:
            resultVideo = cv2.VideoWriter_fourcc(*'MJPG')
            # Writing current processed frame into the video file
            writer = cv2.VideoWriter('result-video.avi', resultVideo, fps = 5,frameSize = (640,480))            
            # Write processed current frame to the file
        writer.write(frame)
        print("write")
        #print(times,frame_counts, class_names, confs)

        yield [frame]#,frame_counts, class_names, confs, class_name, int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]

