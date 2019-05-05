import numpy as np
import os
import tensorflow as tf
import cv2
import math
import timeit

from utils import label_map_util
from utils import visualization_utils as vis_util

FROZEN_INFERENCE_GRAPH_LOC = os.getcwd() + "/inference_graph/frozen_inference_graph.pb"
LABELS_LOC = os.getcwd() + "/" + "label_map.pbtxt"
NUM_CLASSES = 2

#######################################################################################################################
def main():
    print("starting program . . .")

    if not checkIfNecessaryPathsAndFilesExist():
        return

    if tf.__version__ < '1.5.0':
        raise ImportError('Please upgrade your tensorflow installation to v1.5.* or later!')

    # memuat frozen graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(FROZEN_INFERENCE_GRAPH_LOC, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    #memuat label map
    label_map = label_map_util.load_labelmap(LABELS_LOC)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    cap = cv2.VideoCapture('test.mp4')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
 
    # memberikan hasil video output
    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

    total_waktu = 0
    k = 0

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while(cap.isOpened()):
                
                ret, frame = cap.read()
                image_np = frame

                if image_np is None:
                    print("error reading file")
                    break
                
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                image_np_expanded = np.expand_dims(image_np, axis=0)
                # proses mendeteksi objek
                start = timeit.timeit()
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                end = timeit.timeit()

                interval = math.fabs(end-start)

                total_waktu += interval

                threshold = 0.9
        
                objects = []
                
                for index, value in enumerate(classes[0]):
                    object_dict = {}
                    if scores[0, index] > threshold:
                        object_dict[(category_index.get(value)).get('name').encode('utf8')] = \
                                    scores[0, index]
                        objects.append(object_dict)
                        
                stat="safe"
                j=0
                if boxes.any():
                    while j < len(boxes):
                        j+=1
                
                if len(objects) == 0 :
                    i = 0
                else:
                    #menghitung jarak objek
                    height, width = frame.shape[:2]
                    ymax = round(boxes[0][0][2] * height, 0)
                    px = height-ymax
                    # menentukan rumus regresi
                    # dis = round(((0.000007*px**3)-(0.0023*px**2)+(0.5271*px)+43.379), 2)
                    dis = round(((0.0021*px**2)-(0.2998*px)+76.71), 2)
                    
                    str_jarak = 'Distance: '+str(dis)+' cm'
                    font = cv2.FONT_HERSHEY_DUPLEX
                    if dis < 75:
                        str_jarak += ', DANGER!'
                        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 5)
                        cv2.putText(frame, str_jarak, (25, 25), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, str_jarak, (25, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                print(objects, ' ', interval, ' ', k)
                
                vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                                   np.squeeze(boxes),
                                                                   np.squeeze(classes).astype(np.int32),
                                                                   np.squeeze(scores),
                                                                   category_index,
                                                                   use_normalized_coordinates=True,
                                                                   line_thickness=8,
                                                                   min_score_thresh=threshold)
                
                out.write(image_np)
                k+=1
    
    time_average = total_waktu/k

    print(time_average)
    

#######################################################################################################################
def checkIfNecessaryPathsAndFilesExist():

    if not os.path.exists(FROZEN_INFERENCE_GRAPH_LOC):
        print('ERROR: FROZEN_INFERENCE_GRAPH_LOC "' + FROZEN_INFERENCE_GRAPH_LOC + '" does not seem to exist')
        print('was the inference graph exported successfully?')
        return False
    # end if

    if not os.path.exists(LABELS_LOC):
        print('ERROR: the label map file "' + LABELS_LOC + '" does not seem to exist')
        return False
    # end if

    return True
# end function

#######################################################################################################################
if __name__ == "__main__":
    main()
