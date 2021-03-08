import cv2
import numpy as np

vid = cv2.VideoCapture(r'C:\Users\ojask\Desktop\Finding_Lanes\test2.mp4')

def processImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
    [(200,height), (1200,height), (750,300)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask) 
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255,0,0), 10)
    return line_image

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if left_fit:
                left_fit_average = np.average(left_fit, axis=0)
                print(left_fit_average, 'left')
                left_line = make_coordinates(image, left_fit_average)
            if right_fit:
                right_fit_average = np.average(right_fit, axis=0)
                print(right_fit_average, 'right')
                right_line= make_coordinates(image, right_fit_average)
        #return np.array([left_line, right_line])

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*3/5)
    x1 = int(y1 - intercept)/slope
    x1 = int(y2 - intercept)/slope
    return np.array([x1, y1, x2, y2])

while True:
    ret, frame = vid.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed_image = processImage(frame)
    cropped_image = region_of_interest(processed_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(grayFrame, lines)
    line_image = display_lines(cropped_image,lines) 
    combo_image = cv2.addWeighted(grayFrame, .6, line_image, 1, 1)
    cv2.imshow('result', combo_image)
    print(lines)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()