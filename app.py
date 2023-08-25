from flask import Flask, render_template, request, Response, jsonify, json, globals
from flask_cors import CORS, cross_origin
import cv2
import os
import ftplib
import pandas as pd
import urllib.request
import subprocess
import pytz
from pytz import timezone
import time
import uuid
from yolov8_n_opencv import video_detection
from yolov8_n_opencv import *
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from bson.json_util import dumps

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

try:        
    client = MongoClient("mongodb+srv://produzione:YdhlnNyJN3CdyQ2Y@tr2cluster01.2hjuxeq.mongodb.net/")
    database = client["anomalies"]
    anomalycol = database["Anomaly"]
    jobcol = database["Jobs"]
    statuscol = database["SystemStatus2"]
except OperationFailure as e:
    print("Failed to connect to MongoDB")

   
insert = False #variable that control the insertion of anomalies to the database
pause = False
new_job_id = None

def generate_frames_web(videosource):
    count = 0
    global insert, pause
    new_job_id=3
    yolo_output = video_detection(videosource,database,new_job_id)   #calling yolo detection function from yolov8_n_opencv.py
    for detection in yolo_output: #video parameters are called using detection[] from function video_detection()
        ref,buffer=cv2.imencode('.jpg',detection[0])
        frame=buffer.tobytes()
        if pause:
            while pause:  # Pause the algorithm until pause becomes False
                time.sleep(10)
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')
        
        '''ref,buffer=cv2.imencode('.jpg',detection[9])
        frames=buffer.tobytes()
        yield (b'--frames\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frames +b'\r\n')'''

@app.route('/video')
def video():
    videosource="datasets/3.mp4"
    #return frames to display on webpage
    return Response(generate_frames_web(videosource),mimetype='multipart/x-mixed-replace; boundary=frame')

'''@app.route('/preview')
def preview():
    global videosource
    #return frames to display on webpage
    return Response(generate_frames_web(),mimetype='multipart/x-mixed-replace; boundary=frames')'''

#start inserting anomalies to database
@app.route('/start', methods=['GET'])
@cross_origin()
def start():
    print("start called")
    global insert, pause
    insert = True  
    pause = False
    return Response("System Restarted") #replace Response with jsonify to get json output


#stop inserting anomalies to database
@app.route('/stop', methods=['GET'])
@cross_origin()
def stop():
    global insert, pause
    insert = False
    pause = True
    return Response("System Stopped")

@app.route('/live', methods=['POST'])
def live():
    global client, anomalycol, jobcol, statuscol, videosource
    print("live called")
    status = "db connected" if client else "db not Connected"
    try:
        new_job_id=3
        db_return = jobcol.insert_one({"date": datetime.datetime.now(pytz.timezone('Europe/Rome')),"jobid":new_job_id, "stream":"job primo", "location":"job place"})

        #return jsonify(new_job_id)
        data={'jobid':new_job_id}
        response = app.response_class(
            response=json.dumps(data),
            status=200,
            mimetype='application/json'
        )
        return response
    except:
        return Response("Failed to connect to MongoDB")

@app.route('/recorded', methods=['GET'])
def recorded():
    global videosource
    new_job_id=4
    videosource="datasets/3.mp4"
    print(videosource)
    return jsonify(videosource)
   

@app.route('/dbstatus', methods=['GET'])
def get_db_status():
    global client, anomalycol, jobcol, statuscol
    status = "db connected" if client else "db not Connected"
    #store time and connection stop to mongodb collection named SystemStatus
    try:
        statuscol.insert_one({"date": datetime.datetime.now(pytz.timezone('Europe/Rome')),"status": status}) 
        return jsonify({"date": datetime.datetime.now(pytz.timezone('Europe/Rome')),"status": status})
    except:
        return Response("Failed to connect to MongoDB")

@app.route('/')
def index():  
    global insert, pause
    insert = True
    pause = False
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0")