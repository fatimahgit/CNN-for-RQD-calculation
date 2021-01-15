import cv2
import numpy as np
import PIL
from PIL import Image
from pytesseract import image_to_string
import math
def main_extract(img):
    '''
    :param img: tray image
    :return: core tray without the background, depths area, scale area, top text which include the last depth
    '''
    ht, wt, dt = img.shape
    # convert to gray scale to threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    # de-noising using image morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
    # finding the biggest contour which includes the core tray rectangle
    cnts, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(cnts, key=cv2.contourArea)[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    # extract core tray from the original image using the obtained coordinates
    extracted = img[y:y + h, x:x + w]
    # extract depths region located on the left side of the tray
    script_vert = img[y:y + h, 0:x]
    # extract scale region below the core tray
    shift_up = int(math.sqrt((x - x) ** 2 + (ht - (y + h)) ** 2) / 2)
    script_h = img[y + h - shift_up:ht - shift_up, x:x + w]
    #extract the top text region to get the final depth
    script_L_D = img[0:y, x:w]
    # optional for plotting r1 and r2 for later to show the results on the raw image, next to 'final' function
    r1 = x
    r2 = y
    return (extracted, script_vert, script_h, script_L_D, r1, r2)


def isdigit(d):
    try:
        float(d)
        return True
    except ValueError:
        return False

def get_depth(script_vert, script_L_D):
    '''

    :param script_vert: the image region that includes depths of each core row, as displaced on the right side of the image
    :param script_L_D: the image region that includes the last depth of the core tray, needed to calculate core length of
                    the last core row
    :return: a list of all depths
    '''
    # center the script in the middle of the image, currently it is too close to the bottom edge
    script_vert = script_vert[0:script_vert.shape[0], int(0.5 * script_vert.shape[1]): script_vert.shape[1]]
    depths0 = Image.fromarray(script_vert)
    depths0 = depths0.transpose(PIL.Image.ROTATE_270)
    new = Image.new(mode=depths0.mode, size=(depths0.width, 2 * depths0.height), color='white')
    new.paste(depths0)
    #downsample the whole image showed more accurate results
    depths1 = new.resize((int(0.3 * new.width), int(0.3 * new.height)))
    #use pytesseract to detect text, and some post processing to separate and clean the detected depths
    depths = image_to_string(depths1, config='-c tessedit_char_whitelist=.0123456789m')
    depths = depths.replace('m', ' ')
    depths = depths.split(' ')
    depths_list = [d for d in depths if isdigit(d) == True]
    # similarly, get the last depth of the tray from the top text, and add it to other depths
    last_depth = Image.fromarray(script_L_D)
    last_depth = image_to_string(script_L_D, config='-c tessedit_char_whitelist=to.0123456789m')
    last_depth = last_depth.split('to')[-1]
    last_depth = last_depth.split('m')[0]
    depths_list.append(last_depth)
    depths_list = sorted(depths_list, key=float)

    return (depths_list)


def get_scales(script_h):
    '''
    :param script_h: the image region that includes the the scale
    :return: scale for converting pixels to mm
    '''
    # convert to grayscale and threshold
    gray = cv2.cvtColor(script_h, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    # keep only long the horizontal line to detect it in the next step
    kern = np.ones((1, int(0.5 * script_h.shape[1])), np.uint8)
    opening = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kern)

    # detect the horizontal line
    minLineLength = int(0.8 * script_h.shape[1])
    maxLineGap = 0
    lines = cv2.HoughLinesP(opening, 1, np.pi / 180, minLineLength, minLineLength, maxLineGap)

    y_list = sorted([lines[i, 0, 1] for i in range(len(lines))], key=int, reverse=True)
    y = y_list[0]  # this the lowest line (max y) as reverse is true in the sorting of Ys
    scale_image = threshed[y:y + int(script_h.shape[0] / 3), 0:script_h.shape[1]]

    # prepare detection of vertical lines of the scale using erosion and dilation
    kern = np.ones((int(0.5 * scale_image.shape[0]), 1), np.uint8)
    eroded = cv2.erode(scale_image, kern, iterations=2)
    kernd = np.ones((scale_image.shape[0], 1), np.uint8)
    dilated = cv2.dilate(eroded, kernd, iterations=2)

    # detect the lines to get their coordinates and measure the distance in pixles
    minLineLength = int(0.5 * dilated.shape[0])
    maxLineGap = 2
    lines_v = cv2.HoughLinesP(dilated, 1, np.pi / 180, minLineLength, minLineLength, maxLineGap)

    # now, measure the distance in pixels  (between the first two lines)
    line_list = sorted([lines_v[i, 0, 0] for i in range(len(lines_v))], key=int)  # because they are not sorted in the lines array
    distance_pixels = line_list[1] - line_list[0]

    # we need to calculate the distance in mm
    # to get the measurments in texts from the image, crop the image and focus on the measurements part
    scale1 = threshed[y:y + int(threshed.shape[0]), 0:script_h.shape[1]]

    # apply morphology to get rid of the lines as may be interpreted as '1' especially vertical lines
    kernel = np.ones((3, 3), np.uint8)
    scale2 = cv2.morphologyEx(scale1, cv2.MORPH_OPEN, kernel)
    # the image is inverted threshold, inverse it here to have white background and black text
    scale2 = cv2.bitwise_not(scale2)
    scale_img = Image.fromarray(scale2)
    scale_list = image_to_string(scale_img, lang='eng', config='--psm 7 -c tessedit_char_whitelist=0123456789')
    scale_list = scale_list.split(' ')
    scale_list = sorted(scale_list, key=int)
    # find distance in mm
    distance_mm = int(scale_list[2]) - int(scale_list[1])
    # find the scale to convert from pixels to mm
    pixels_eq_10cm = 100 * distance_pixels / distance_mm
    return (int(pixels_eq_10cm))