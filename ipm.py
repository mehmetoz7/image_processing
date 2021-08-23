import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import threading
import logging
import time
import math

from myLine import myLine
from time import sleep
from PIL import Image
from pynput import keyboard
from utils import perspective, Plane, load_camera_params, bilinear_sampler, warped

image = cv2.cvtColor(cv2.imread('1.png'), cv2.COLOR_BGR2RGB)
interpolation_fn = bilinear_sampler  # or warped
TARGET_H, TARGET_W = 800, 800
INCREMENT_COEFF = 0.001
PARALLELISM_TOLERANCE = 1 #comparison level for detecting paralelism 

def ipm_from_parameters(image, xyz, K, RT, interpolation_fn):
    # Flip y points positive upwards
    xyz[1] = -xyz[1]
    P = K @ RT
    pixel_coords = perspective(xyz, P, TARGET_H, TARGET_W)
    image2 = interpolation_fn(image, pixel_coords)
    return image2.astype(np.uint8)

def warp(pitch):
    # Derived method
    plane = Plane(0, -40, 0, 0, 0, 0, TARGET_H, TARGET_W, 0.1)
    # Retrieve camera parameters
    extrinsic, intrinsic = load_camera_params('camera.json')

    # Apply perspective transformation
    warped1 = ipm_from_parameters(image, plane.xyz, intrinsic, extrinsic, interpolation_fn)

    fig, ax1 = plt.subplots(1,1)
    im = ax1.imshow(warped1)
    val = 'IPM with ' + str(pitch) + ' Pitch'
    ax1.set_title('IPM with ' + str(pitch) + ' Pitch')
    plt.tight_layout()
    plt.savefig('org.png')    
    gray = cv2.cvtColor(warped1, cv2.COLOR_BGR2GRAY) 

    cv2.imwrite('gray.png', gray)    

def filter_out_dark_pixels():
    im = Image.open('gray.png')
    px = im.load()

    for i in range(im.width): 
       for j in range(im.height):    
           if (px[i,j] < 150):
               px[i,j] = 0

    im.save('threshold_filter_gray.png')

def distance(x1, y1, x2, y2):
    d = math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
    return d

def get_lines():
    det_lines = []
    rm_lines = []
    threshold_filter_gray = cv2.imread('threshold_filter_gray.png')
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(threshold_filter_gray,(kernel_size, kernel_size),0)
    low_threshold = 50
    high_threshold = 150
    canny_out = cv2.Canny(blur_gray, low_threshold, high_threshold)
    cv2.imwrite('canny_out.png',canny_out)
    
    #get lines
    lines = cv2.HoughLines(canny_out, 1, np.pi / 180, 240)
    
    if len(lines) == 0:
        return 0

    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b)) # x1 stores the rounded off value of (r * cos(theta) - 1000 * sin(theta))
        y1 = int(y0 + 1000 * (a)) # y1 stores the rounded off value of (r * sin(theta)+ 1000 * cos(theta))
        x2 = int(x0 - 1000 * (-b)) # x2 stores the rounded off value of (r * cos(theta)+ 1000 * sin(theta))
        y2 = int(y0 - 1000 * (a)) # y2 stores the rounded off value of (r * sin(theta)- 1000 * cos(theta))
        ll = myLine(x1, y1, x2, y2)
        det_lines.append(ll)
        cv2.line(threshold_filter_gray, (x1, y1), (x2, y2), (255, 0, 0), 3)
        #print("x1: " + str(x1) + "\ty1:" + str(y1)+ "\tx2:" + str(x2)+ "\ty2:" + str(y2))

    #find duplicate lines
    lineCount = len(det_lines)
    #print("\nlineCount: " + str(lineCount))
    if(lineCount >= 2):
        i = 0
        while i<lineCount:  
            j = i + 1
            while j<lineCount:
                #print(str(i) + " " + str(j))                
                #print(str(det_lines[i].x1) + "\t" + str(det_lines[i].y1)+ "\t" + str(det_lines[i].x2)+ "\t" + str(det_lines[i].y2))
                #print(str(det_lines[j].x1) + "\t" + str(det_lines[j].y1)+ "\t" + str(det_lines[j].x2)+ "\t" + str(det_lines[j].y2))
                dist_start = distance(det_lines[i].x1, det_lines[i].y1, det_lines[j].x1, det_lines[j].y1)
                dist_end = distance(det_lines[i].x2, det_lines[i].y2, det_lines[j].x2, det_lines[j].y2)
                #print("d1 " + str(dist_start))
                #print("d2 " + str(dist_end))
                if(dist_start < 30 and dist_end < 30):
                    #print("****remove " + str(i))
                    if(i not in rm_lines):
                        rm_lines.append(i)
                j = j + 1
            i = i + 1
    
    #remove duplicate lines
    #print("rm_lines" + str(rm_lines))
    while len(rm_lines) > 0:
        det_lines.pop(rm_lines[0])
        rm_lines.pop(0)
        rm_lines = [x - 1 for x in rm_lines]
        #print("rm_lines" + str(rm_lines))

    #draw plot
    for i in range(len(det_lines)):
        cv2.line(threshold_filter_gray, (det_lines[i].x1, det_lines[i].y1), (det_lines[i].x2, det_lines[i].y2), (0, 0, 255), 3)
    
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 900, 900)
    cv2.imshow('image', threshold_filter_gray)
    cv2.imwrite('lines.png', threshold_filter_gray)

    lineCount = len(det_lines)
    if(lineCount >= 2):
        i = 0
        while i<lineCount:  
            j = i + 1
            while j<lineCount:
                #print(str(i) + " " + str(j))                
                #print(str(det_lines[i].x1) + "\t" + str(det_lines[i].y1)+ "\t" + str(det_lines[i].x2)+ "\t" + str(det_lines[i].y2))
                #print(str(det_lines[j].x1) + "\t" + str(det_lines[j].y1)+ "\t" + str(det_lines[j].x2)+ "\t" + str(det_lines[j].y2))
                dist_start = distance(det_lines[i].x1, det_lines[i].y1, det_lines[j].x1, det_lines[j].y1)
                dist_end = distance(det_lines[i].x2, det_lines[i].y2, det_lines[j].x2, det_lines[j].y2)
                #print("d1 " + str(dist_start))
                #print("d2 " + str(dist_end))
                if(abs(dist_start - dist_end) < PARALLELISM_TOLERANCE):
                    return 1
                j = j + 1
            i = i + 1
    #print("\n***************************************\n")
    return 0  


if __name__ == '__main__':
    a_file = open("camera.json", "r")
    json_object = json.load(a_file)
    pitch = json_object['pitch']
    a_file.close()

    diff_min = 1000
    while True:
        print("pitch: " + str(pitch))
        warp(pitch)
        filter_out_dark_pixels()

        if(get_lines() == 1):
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break
        else:
            cv2.waitKey(1)
        
        pitch = pitch + INCREMENT_COEFF
        json_object['pitch'] = pitch
        a_file = open("camera.json", "w")
        json.dump(json_object, a_file)
        a_file.close()

    print("\n\n\nexit")   

