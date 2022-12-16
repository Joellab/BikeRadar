#!/usr/bin/env python3
# coding=utf-8

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
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

SERVER_URL = "http://ilibrary.api.dev.joellab.com"
TOTAL_THREAD = 0
CMD = "INIT"
ACTION = "INIT"
CURRENT_JOB = "INIT"

def cmdQuery():
    #TODO: Use WebSocket instead of query
    global CURRENT_JOB
    prevJob = "NONE"
    while True:
        try:
            url = SERVER_URL + '/api/comm/cmd'
            r = requests.get(url, timeout=3)
            r = r.json()
        
            cmdLock.acquire()
            CMD = r.get("data")
            cmdLock.release()

            if CMD != prevJob:
                if CMD.split("|")[0] == "NEWJOB":
                    CURRENT_JOB = CMD.split("|")[1]
                    ACTION = "NAVPOINT"
                elif CMD.split("|")[0] == "STOP":
                    CURRENT_JOB = "INIT"
                    ACTION = "INIT"
                prevJob = CMD
                rospy.loginfo("[ILibrary] New command received: " + CMD)
        except:
            rospy.loginfo("[ILibrary] Command query failed")
        time.sleep(3)


def waitForJob():
    global ACTION
    global CURRENT_JOB
    while ACTION == "STANDBY":
        try:
            url = SERVER_URL + '/api/comm/job'
            params = {"current": CURRENT_JOB}
            r = requests.get(url, params, timeout=3)
            r = r.json()
            if r.get("data").get("action") != "STANDBY":
                ACTION = r.get("data").get("action")
                actionProcess(r)
        except:
            rospy.loginfo("[ILibrary] Job query failed 2")
        time.sleep(3)

def getNextJob():
    global ACTION
    global CURRENT_JOB
    try:
        url = SERVER_URL + '/api/comm/job'
        params = {"current": CURRENT_JOB}
        r = requests.get(url, params, timeout=3)
        r = r.json()
        ACTION = r.get("data").get("action")
        CURRENT_JOB = r.get("data").get("uuid")
        print(ACTION)
        actionProcess(r)

    except:
        rospy.loginfo("[ILibrary] Job query failed 1")

def navPoint(posiX, posiY, posiZ, oriX, oriY, oriZ, oriW):
    navGoal = MoveBaseGoal()
    navGoal.target_pose.header.frame_id="map"
    navGoal.target_pose.header.frame_id="map"
    navGoal.target_pose.pose.position.x = float(posiX)
    navGoal.target_pose.pose.position.y = float(posiY)
    navGoal.target_pose.pose.position.z = float(posiZ)
    navGoal.target_pose.pose.orientation.x = float(oriX)
    navGoal.target_pose.pose.orientation.y = float(oriY)
    navGoal.target_pose.pose.orientation.z = float(oriZ)
    navGoal.target_pose.pose.orientation.w = float(oriW)
    actionClient.send_goal(navGoal)
    rospy.loginfo("[ILibrary] Nav Start")
    actionClient.wait_for_result()
    rospy.loginfo("[ILibrary] Nav Done")
    time.sleep(3)
    getNextJob()


def actionProcess(r):
    global ACTION
    if ACTION == "STANDBY":
        waitForJob()
    elif ACTION == "NAVPOINT":
        navPoint(r.get("data").get("posiX"), r.get("data").get("posiY"), r.get("data").get("posiZ"), r.get("data").get("oriX"), r.get("data").get("oriY"), r.get("data").get("oriZ"), r.get("data").get("oriW"))
    else:
        rospy.loginfo("[ILibrary] Unknow action")

def init():
    cmdQueryThread = threading.Thread(target=cmdQuery)
    cmdQueryThread.start()
    getNextJob()

if __name__ == "__main__":
    rospy.loginfo("[ILibrary] Service Start")
    rospy.init_node("simple_goal")
    actionClient = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    actionClient.wait_for_server()
    cmdLock = threading.Lock()
    init()
    rospy.loginfo("[ILibrary] Service Exit")