import cv2
import numpy as np
import math
import os
import time

FOV = 68.5

cap = cv2.VideoCapture(1)

lower_retroreflective = np.array([0, 0, 112])
upper_retroreflective = np.array([150, 179, 255])
kernel = np.ones((5,5),np.uint8)

lower_retroreflective2 = np.array([0, 150, 112])
upper_retroreflective2 = np.array([100, 255, 255])
def get_images(directory=None):
   
    if directory == None:
        directory = os.getcwd() # Use working directory if unspecified
        
    image_list = []
    
    directory_list = os.listdir(directory) # Get list of files
    for entry in directory_list:
        absolute_filename = os.path.join(directory, entry)
        try:
            image = cv2.imread(absolute_filename)
            image_list += [image]
        except IOError:
            pass # do nothing with errors tying to open non-images
    return image_list

def calculate_coverage_area_score(coverage_area):
    return math.exp(-math.pow(coverage_area-1/3.0, 2) / float(2 * math.pow(3, 2))) * 100

def calculate_particle_area(image, x_pos, y_pos, width, height):
    total = 0
    sub_rows = image[y_pos:y_pos+height + 1]
    for row in sub_rows:
        sub_cols = row[x_pos:x_pos + width + 1]
        total += cv2.sumElems(sub_cols)[0]
    return total

def calculate_coverage_area(particle_area, bounding_box_area):
    return particle_area / float(bounding_box_area)

def apply_morphologyEx(mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def calculate_aspect_ratio_score(width, height):
    ratio = width / float(height)
    dar = 1.6 - ratio
    return -200 * abs(dar) + 100

def pixel_to_aiming(x, y, resolution_x, resolution_y):
    aiming_x = (x - resolution_x / 2.0) / (resolution_x / 2.0)
    aiming_y = (y - resolution_y / 2.0) / (resolution_y / 2.0)
    return (aiming_x, -aiming_y)
    

images = get_images('/home/kyle/Downloads/RealFullField')

for frame in images:
##    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_retroreflective, upper_retroreflective)
    mask = apply_morphologyEx(mask)
    mask2 = cv2.inRange(hsv, lower_retroreflective2, upper_retroreflective2)
    mask2 = apply_morphologyEx(mask2)
    mask = cv2.bitwise_or(mask, mask2)
    cv2.imshow("hsv", hsv)
    cv2.imshow("mask", mask)
    cv2.imshow("m2", mask2)
    contours, h = cv2.findContours(mask, 1, 2)
    for cnt in contours:
##        rect = cv2.minAreaRect(cnt)
##        box = cv2.boxPoints(rect)
##        box = np.int0(box)
##        cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
        x,y,w,h = cv2.boundingRect(cnt)
        coverage_area = calculate_coverage_area(calculate_particle_area(mask, x, y, w, h), w * h)
        score = calculate_coverage_area_score(coverage_area)
##        aspect_ratio_score = calculate_aspect_ratio_score(w, h)
        if score >= 70:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            print pixel_to_aiming(x + w/2.0, y + h/2.0, len(frame[0]), len(frame))
    cv2.imshow("frame", frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    time.sleep(0.2)
cv2.destroyAllWindows()
cap.release()
