import cv2
import numpy as np


def x_cord_contours(contours):
    M = cv2.moments(contours)
    if M['m10'] != 0 and M['m00'] != 0:
        x_moments = M['m10'] / M['m00']
    else:
        x_moments = 0
    return x_moments


def product(val):
    prod = val[0] * val[1]
    return prod


def erect_plate(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 120)

    contours, ret = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(plate, contours, -1, (0, 255, 0), 2)
    mbox = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        mbox.append(box)

    sorted_boxes = sorted(mbox, key=cv2.contourArea, reverse=True)
    roi = sorted_boxes[0]
    i, j, k, l = roi

    order = []
    tempX = [abs(product(i)), abs(product(j)), abs(product(k)), abs(product(l))]
    print(len(tempX))
    for n in range(4):
        if tempX is []:
            continue
        temp = min(tempX, default='Empty')
        if temp == abs(product(i)):
            order.append(i)
            tempX.remove(abs(product(i)))
        if temp == abs(product(j)):
            order.append(j)
            tempX.remove(abs(product(j)))
        if temp == abs(product(k)):
            order.append(k)
            tempX.remove(abs(product(k)))
        if temp == abs(product(l)):
            order.append(l)
            tempX.remove(abs(product(l)))

    if order[1][0] < order[2][0]:
        temp = order[1]
        order[1] = order[2]
        order[2] = temp

    a, b, c, d = order
    points_A = np.float32([a, b, c, d])
    a1 = [0, 0]
    b1 = [1000, 0]
    c1 = [0, 500]
    d1 = [1000, 500]
    points_B = np.float32([a1, b1, c1, d1])

    M = cv2.getPerspectiveTransform(points_A, points_B)
    wraped = cv2.warpPerspective(plate, M, (1000, 500))
    return wraped


def uneven_lengths(lengths):
    ad_lengths = []
    for a in lengths:
        for b in lengths:
            if lengths.index(a) == lengths.index(b):
                continue
            if a == b:
                a = a + 1
        ad_lengths.append(a)
    return ad_lengths


def box_check(contours, shape):
    # Ratio Check
    min_ratio = 1.5
    max_ratio = 5

    # Vertical Distance from plate center check
    hph = shape[0] / 2  # half of plate height
    max_vd = hph / 3   # Maximum vertical distance


    max_angle_diff = 0
    max_area_diff = 0
    max_height_diff = 0
    max_width_diff = 0

    i = 0
    contour_dict = []

    # First tier of filtering
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cy = y + (h / 2)
        dfc = abs(hph - cy)     # Distance from plate center
        ratio = h / w

        if min_ratio < ratio < max_ratio and dfc < max_vd:
            contour_dict.append({'contour': cnt,
                                 'idx': i,
                                 'x': x,
                                 'y': y,
                                 'w': w,
                                 'h': h,
                                 'cx': x + (w / 2),
                                 'cy': y + (h / 2)})
        i += 1

    # Second tier of filtering
    max_f = 1.1     # maximum division factor
    min_f = 0.9     # minimum division factor
    heights = []
    for dict in contour_dict:
        heights.append(dict['h'])

    match_idx = []
    for idx, h1 in enumerate(heights):
        temp_idx = []
        temp_idx.append(idx)
        for idy, h2 in enumerate(heights):
            if idx == idy:
                continue
            frac = h1 / h2
            if min_f < frac < max_f:
                temp_idx.append(idy)
        if len(temp_idx) > 5:
            match_idx.append(temp_idx)

    return contour_dict, match_idx


def segment1(img):
    shape = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, tresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilation = cv2.dilate(tresh, kernel, iterations=1)
    contours, hirachy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # Sort Contours
    sorted_contours = sorted(contours, key=x_cord_contours, reverse=False)
    contour_dict, match_idx = box_check(sorted_contours, shape)
    for dict in contour_dict:
        x = dict['x']
        y = dict['y']
        w = dict['w']
        h = dict['h']
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img, contour_dict, match_idx


def segment2(img):
    shape = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # bit_blur = cv2.bilateralFilter(gray, 10, 20, 100, borderType=cv2.BORDER_CONSTANT)
    # kernel = np.array([[-1, -1, -1],
    #                    [-1, 9, -1],
    #                    [-1, -1, -1]])
    # sharpened = cv2.filter2D(bit_blur, cv2.CV_8UC3, kernel)
    # blur = cv2.GaussianBlur(sharpened, (5, 5), 0)
    ret, tresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # dilation = cv2.dilate(tresh, kernel, iterations=1)
    contours, hirachy = cv2.findContours(tresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=x_cord_contours, reverse=False)
    contour_dict, match_idx = box_check(sorted_contours, shape)
    for dict in contour_dict:
        x = dict['x']
        y = dict['y']
        w = dict['w']
        h = dict['h']
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img, contour_dict, match_idx


def segment3(img):
    shape = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # bit_blur = cv2.bilateralFilter(gray, 10, 20, 100, borderType=cv2.BORDER_CONSTANT)
    # kernel = np.array([[-1, -1, -1],
    #                    [-1, 9, -1],
    #                    [-1, -1, -1]])
    # sharpened = cv2.filter2D(bit_blur, cv2.CV_8UC3, kernel)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 20, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilation = cv2.dilate(edges, kernel, iterations=1)
    contours, hirachy = cv2.findContours(dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=x_cord_contours, reverse=False)
    contour_dict, matched_idx = box_check(sorted_contours, shape)
    for dict in contour_dict:
        x = dict['x']
        y = dict['y']
        w = dict['w']
        h = dict['h']
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img, contour_dict, matched_idx


def padding(img, ratio):
    gray = [150, 150, 150]
    h = img.shape[0]
    w = img.shape[1]
    wp = h/ratio
    pad = int((wp - w)/2)
    img = cv2.copyMakeBorder(img,0,0,pad,pad,cv2.BORDER_CONSTANT,value=gray)
    return img


# def segment3(img):


# img = cv2.imread(r'C:\Users\Hp\Desktop\Python\Datasets\Test_Plate1.jpg')
# image = erect_plate(img)
# image_roi = segment1(image.copy())
# cv2.imshow('Original Image', img)
# cv2.waitKey(0)
# cv2.imshow('Erected', image)
# cv2.waitKey(0)
# cv2.imshow('ROI', image_roi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
