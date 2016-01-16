import cv2
import numpy as np
import math

cap = cv2.VideoCapture(1)

lower_retroreflective = np.array([0, 0, 112])
upper_retroreflective = np.array([int(55/180.0 * 255), 179, 255])
kernel = np.ones((5,5),np.uint8)


def calculate_score(coverage_area):
    return math.exp(-math.pow(x-1/3.0, 2) / float(2 * math.pow(0.5, 2))) * 100

def calculate_particle_area(image, x_pos, y_pos, width, height):
    total = 0
    for row in range(y_pos, y_pos+height):
        for col in range(x_pos, x_pos + width):
            total += image[row][col]
    return float(total)

def calculate_coverage_area(particle_area, bounding_box_area):
    return particle_area / float(bounding_box_area)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_retroreflective, upper_retroreflective)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("hsv", hsv)
    cv2.imshow("mask", mask)
    contours, h = cv2.findContours(mask, 1, 2)
    count = 0
    for cnt in contours:
##        rect = cv2.minAreaRect(cnt)
##        box = cv2.boxPoints(rect)
##        box = np.int0(box)
##        cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        coverage_area = calculate_coverage_area(calculate_particle_area(mask, x, y, w, h), w * h)
        print calculate_score(coverage_area)
        count += 1
        if count > 4:
            break
    cv2.imshow("frame", frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()
cap.release()
