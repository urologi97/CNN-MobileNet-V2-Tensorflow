# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import timeit
import cv2
import os
import numpy as np
import tensorflow as tf
import sys
import math

sys.path.append("..")

from utils import label_map_util

MODEL_NAME = 'inference_graph'

CWD_PATH = os.getcwd()

PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

PATH_TO_LABELS = os.path.join(CWD_PATH,'label_map.pbtxt')

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
 
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 15
rawCapture = PiRGBArray(camera, size=(640, 480))

time.sleep(0.1)

def main():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    file = open(timestr+".txt", "w")
    start_graph = timeit.timeit()
    #load graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)
    end_graph = timeit.timeit()

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    print('GRAPH LOADED IN ',end_graph-start_graph,'S')

    total_waktu = 0
    k = 0

    try:
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

            image = frame.array
            
            frame_expanded = np.expand_dims(image, axis=0)

            start = timeit.timeit()
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded})
            end = timeit.timeit()

            interval = math.fabs(end-start)
            total_waktu += interval

            threshold = 0.9
            dis = 0
    
            objects = []
            
            for index, value in enumerate(classes[0]):
                object_dict = {}
                if scores[0, index] > threshold:
                    object_dict[(category_index.get(value)).get('name').encode('utf8')] = \
                                scores[0, index]
                    objects.append(object_dict)
                    
            stat="Safe"
            j=0
            if boxes.any():
                while j < len(boxes):
                    j+=1
            
            if len(objects) == 0 :
                i = 0
            else:
                i += 1
                height, width = image.shape[:2]
                ymax = round(boxes[0][0][2] * height, 0)
                px = height-ymax
                # dis = round(((0.000007*px**3)-(0.0023*px**2)+(0.5271*px)+43.379), 2)
                dis = round(((0.0021*px**2)-(0.2998*px)+76.71), 2)
                str_jarak = 'Distance: '+str(dis)+' cm '
                font = cv2.FONT_HERSHEY_DUPLEX
                if dis < 75:
                    stat = 'DANGER!'
                    str_jarak = ', ' + stat
                    cv2.rectangle(image, (0, 0), (width, height), (0, 0, 255), 5)
                    cv2.putText(image, str_jarak, (25, 25), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                else:
                    cv2.putText(image, str_jarak, (25, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            strTxt = str(objects)+ str(i)+'detected in '+str(interval) + 's, Distace: ' + str(dis) + 'cm, ' + str(stat)
            print(strTxt)
            file.write(str(interval))
            rawCapture.truncate(0)
            k+=1
    except KeyboardInterrupt:
        exTimeAvg = total_waktu/k
        print(exTimeAvg)
        file.write("time avg : "+ str(exTimeAvg))
        pass

if __name__ == '__main__':
    main()