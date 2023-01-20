import socket
import struct
import pickle
import cv2
import numpy as np
import tensorflow as tf
import time
from Character_Segment import erect_plate, segment1, segment2, segment3, padding

PORT = 9999
# HOST_IP = '***.***.**.**' # input IP of server computer
HOST_IP = '192.168.43.78'

# create socket client
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

classifier = tf.keras.models.load_model('Main2_CR_CNN_50_Epochs.h5')
Y = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
     'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

whT = 320
confThreshold = 0.5
nmsThreshold = 0.3

classesFile = 'coco_custom.names'
className = []
with open(classesFile, 'rt') as f:
    className = f.read().rstrip('\n').split('\n')

modelConfiguration = 'yolov4-custom.cfg'
modelWeights = 'custom.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    obj = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'{className[classIds[i]].upper()} {int(confs[i] * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        obj.append(img[y: y + h, x: x + w])
    return obj


def from_feed():
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('Test_Video.mp4')
    blank = np.zeros((250, 500))
    print('Connecting to server... ')
    client.connect((HOST_IP, PORT))
    plate_number = ""
    soc_plate = blank
    soc_plate_roi = blank

    while True:
        # time.sleep(.05)
        ret, image = cap.read()
        cv2.waitKey(1)

        if ret is True:

            blob = cv2.dnn.blobFromImage(image, 1 / 255, (whT, whT), [0, 0, 0], crop=False)
            net.setInput(blob)

            layerNames = net.getLayerNames()
            outLayersIndex = net.getUnconnectedOutLayers()
            # print(layerNames, outLayersIndex, sep='\n')
            try:
                outLayers = [layerNames[i[0] - 1] for i in outLayersIndex]
            except IndexError:
                outLayers = [layerNames[i - 1] for i in outLayersIndex]

            outputs = net.forward(outLayers)
            plates = findObjects(outputs, image)

            cv2.imshow('Camera Feed', image)
            # Send Video
            soc_vid = cv2.resize(image, (480, 320), cv2.INTER_AREA)

            if plates:
                for plate in plates:
                    if type(plate) == np.ndarray and plate.size > 0:

                        ### Start
                        functions = [segment1, segment2, segment3]

                        for segment in functions:
                            # cv2.imshow('License_Plate', plate)
                            # img = erect_plate(plate)
                            if plate.shape[1] < 400:
                                plate = cv2.resize(plate, (900, 450), cv2.INTER_AREA)
                            # cv2.imshow('Licensed Plate', plate)
                            # Send image
                            soc_plate = cv2.resize(plate, (500, 250), cv2.INTER_AREA)

                            img_roi, contour_dict, match_idx = segment(plate.copy())
                            if match_idx:
                                break

                        if not match_idx:
                            continue

                        count = 0
                        full_char = []
                        for idx in match_idx[0]:
                            dict = contour_dict[idx]
                            x = dict['x'] - 5
                            y = dict['y'] - 5
                            w = dict['w'] + 10
                            h = dict['h'] + 10
                            char = plate[int(y): int(y + h), int(x): int(x + w)]
                            if char.size > 0:
                                pass
                            else:
                                continue

                            # Padding
                            ratio = 40 / 32
                            char_pad = padding(char, ratio)

                            r_char = cv2.resize(char_pad, (32, 40), cv2.INTER_AREA)
                            g_char = cv2.cvtColor(r_char, cv2.COLOR_BGR2GRAY)
                            g_char = g_char / 255.0
                            feed = g_char.reshape(1, 40, 32, 1)
                            out = classifier.predict(feed)  # , 1, verbose=0)[0]
                            result_idx = np.argmax(out)
                            result = Y[result_idx]
                            cv2.rectangle(plate, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(plate, result, (x, y), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
                            # cv2.imshow('Characters', plate)
                            # Send image
                            soc_plate_roi = cv2.resize(plate, (500, 250), cv2.INTER_AREA)

                            # print(result)
                            full_char.append(result)
                            # cv2.waitKey(0)

                        read = "".join(full_char)
                        if plate_number != read:
                            plate_number = read
                            print("The License Plate Number is: " + plate_number)
                        
                        # Send Information
                        strut = {'video': soc_vid, 'plate': soc_plate, 'plate roi': soc_plate_roi, 'plate number': plate_number}
                        p_img = pickle.dumps(strut)
                        message = struct.pack("Q", len(p_img)) + p_img
                        client.sendall(message)

            else:
                soc_plate = blank
                soc_plate_roi = blank
                plate_number = "Null"
                strut = {'video': soc_vid, 'plate': soc_plate, 'plate roi': soc_plate_roi, 'plate number': plate_number}
                p_img = pickle.dumps(strut)
                message = struct.pack("Q", len(p_img)) + p_img
                client.sendall(message)

                cv2.waitKey(1)

    cv2.destroyAllWindows()


from_feed()
