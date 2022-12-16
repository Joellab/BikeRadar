#!/usr/bin/env python3
# coding:utf-8

import rospy
import cv2
import subprocess
import base64
import requests
import json
import time
import threading
import sys
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import copy

DELAY_TIME = 0
DELAY_TIME_MAX = 50
TOTAL_THREAD = 0
SERVER_URL = "http://ilibrary.api.dev.joellab.com"

def showVideo(name, img, zoom):
    h, w, c = img.shape
    resizeWidth = int(w / zoom)
    resizeHeight = int(h / zoom)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, resizeWidth, resizeHeight)
    cv2.imshow(name, img)

def camProcess(data):
    global DELAY_TIME
    cvImg = bridge.imgmsg_to_cv2(data, "bgr8")
    imgInfo = cvImg.shape
    size = (imgInfo[1], imgInfo[0])
    showVideo("Main Camera", cvImg, 5)
    
    if(TOTAL_THREAD >=3):
        DELAY_TIME_MAX = 100
    else:
        DELAY_TIME_MAX = 50
    
    if(DELAY_TIME >= DELAY_TIME_MAX):
        imgAnalyse = threading.Thread(target=analyseQRCode, args=(cvImg,))
        imgAnalyse.start()
        DELAY_TIME = 0
    cv2.waitKey(1)
    DELAY_TIME += 1

def imgEncode(img):
    params = [cv2.IMWRITE_JPEG_QUALITY, 20]
    base64Str = cv2.imencode('.jpg', img, params)[1].tostring()
    base64Str = base64.b64encode(base64Str)
    return base64Str

def analyseQRCode(img):

    #imgBase64 = data['imgBase64']
    #increaseUnit = data['increaseUnit']
    #currentPosiHead = data['currentPosiHead']
    #imgBase64Dec = base64.b64decode(imgBase64)
    #imgArray = np.frombuffer(imgBase64Dec,np.uint8)
    #img=cv2.imdecode(imgArray,cv2.COLOR_BGR2RGB)

    global TOTAL_THREAD
    TOTAL_THREAD += 1
    currThread = copy.deepcopy(TOTAL_THREAD)

    rospy.loginfo("[ILibrary] Thread-" + str(currThread) + " Analysing...")
    detector = cv2.wechat_qrcode_WeChatQRCode("/home/public/ILibrary/ROS/src/wpr_simulation/scripts/model/detect.prototxt", "/home/public/ILibrary/ROS/src/wpr_simulation/scripts/model/detect.caffemodel", "/home/public/ILibrary/ROS/src/wpr_simulation/scripts/model/sr.prototxt", "/home/public/ILibrary/ROS/src/wpr_simulation/scripts/model/sr.caffemodel")
    imgHeight = img.shape[0]
    imgLength = img.shape[1]
    currentPosiHead = 0
    increaseUnit = 10
    currentPosiTail = currentPosiHead
    resList = []

    while currentPosiHead < imgLength:
        currentPosiTail += increaseUnit
        if currentPosiTail > imgLength:
            currentPosiTail = imgLength
        croppedImg = img[0:imgHeight, currentPosiHead:currentPosiTail]
        msg, border = detector.detectAndDecode(croppedImg)
        if len(msg) == 0:
            if currentPosiTail == imgLength:
                currentPosiHead = currentPosiTail
        else:
            currentPosiTail = currentPosiHead + int(border[0][1][0])
            currentPosiHead = currentPosiTail
            resList.append(msg[0])

    scanResult = json.dumps(resList, ensure_ascii=False) 

    rospy.loginfo("[ILibrary] Thread-" + str(currThread) + " Done" + "\n" + scanResult)

    if len(resList) > 0:
        try:
            url = SERVER_URL + '/api/comm/scan'
            data = scanResult
            r = requests.post(url, json=data, timeout=3)
        except:
            rospy.loginfo("[ILibrary] Thread-" + str(currThread) + ' Sync scan result timeout')

    try:
        url = SERVER_URL + '/api/comm/scan/img'
        value = {
            "img" : imgEncode(img)
        }
        r = requests.post(url, json=value, timeout=3)
    except:
        rospy.loginfo("[ILibrary] Thread-" + str(currThread) + " Image upload timeout")

    TOTAL_THREAD -= 1
    return scanResult

def init():
    rospy.loginfo("[ILibrary] Service Start")
    rospy.init_node('img_process_node', anonymous=True)
    rospy.Subscriber('/kinect2/hd/image_color_rect', Image, camProcess)
    rospy.spin()
    rospy.loginfo("[ILibrary] Service Exit")


if __name__ == '__main__':
    bridge = CvBridge()
    init()


