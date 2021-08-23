import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import threading
import logging
import time

from time import sleep
from PIL import Image
from pynput import keyboard
from utils import perspective, Plane, load_camera_params, bilinear_sampler, warped

image = cv2.cvtColor(cv2.imread('1.png'), cv2.COLOR_BGR2RGB)
interpolation_fn = bilinear_sampler  # or warped
TARGET_H, TARGET_W = 800, 800
quanta = 0.001

def ipm_from_parameters(image, xyz, K, RT, interpolation_fn):
    # Flip y points positive upwards
    xyz[1] = -xyz[1]
    P = K @ RT
    pixel_coords = perspective(xyz, P, TARGET_H, TARGET_W)
    image2 = interpolation_fn(image, pixel_coords)
    return image2.astype(np.uint8)

def ipm_from_opencv(image, source_points, target_points):
    # Compute projection matrix
    M = cv2.getPerspectiveTransform(source_points, target_points)
    # Warp the image
    warped = cv2.warpPerspective(image, M, (TARGET_W, TARGET_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return warped

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

def get_lines():
    m = {""}
    threshold_filter_gray = cv2.imread('threshold_filter_gray.png')
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(threshold_filter_gray,(kernel_size, kernel_size),0)
    low_threshold = 50
    high_threshold = 150
    canny_out = cv2.Canny(blur_gray, low_threshold, high_threshold)
    cv2.imwrite('canny_out.png',canny_out)
    
    lines = cv2.HoughLines(canny_out, 1, np.pi / 180, 250)
    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # x1 stores the rounded off value of (r * cos(theta) - 1000 * sin(theta))
        x1 = int(x0 + 1000 * (-b))
        # y1 stores the rounded off value of (r * sin(theta)+ 1000 * cos(theta))
        y1 = int(y0 + 1000 * (a))
        # x2 stores the rounded off value of (r * cos(theta)+ 1000 * sin(theta))
        x2 = int(x0 - 1000 * (-b))
        # y2 stores the rounded off value of (r * sin(theta)- 1000 * cos(theta))
        y2 = int(y0 - 1000 * (a))
        cv2.line(threshold_filter_gray, (x1, y1), (x2, y2), (0, 0, 255), 1)
        m.add((y2 - y1)/(x2-x1))

    cv2.imshow('image', threshold_filter_gray)
    cv2.imwrite('lines.png', threshold_filter_gray)
    print(m)
    k = cv2.waitKey(0)
    return m

if __name__ == '__main__':
    a_file = open("camera.json", "r")
    json_object = json.load(a_file)
    pitch = json_object['pitch']
    a_file.close()

    diff_min = 1000
    while True:
        print("pitch: " + str(pitch))
        warp(pitch)
        sleep(0.1)
        filter_out_dark_pixels()
        sleep(0.1)
        m = get_lines()        
        m.remove("")
        
        print(m)

        if(len(m) == 2):
            diff = abs(list(m)[0] - list(m)[1])            
            print("diff : " + str(diff))
            #if(diff_min > diff ):
            #    diff_min = diff
            #else:
            #    break

        print()

        pitch = pitch + 0.0005
        json_object['pitch'] = pitch
        a_file = open("camera.json", "w")
        json.dump(json_object, a_file)
        a_file.close()      

        sleep(0.1)
        