# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:42:25 2019

@author: liuyizhuang
"""

import numpy as np
import cv2
import os
import sys
import tensorflow as tf
import time
from PIL import Image

def file_name(path):
    for root,dirs,files in os.walk(path):
        return files

def run(filename,sess,image_tensor,boxes,scores,classes,num_detections):
    image = Image.open(filename)
    image_np = np.array(image).astype(np.uint8)
    image_exp = np.expand_dims(image_np, axis=0)
    t1 = time.time() 
    (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_exp})
    print(scores)
    print(classes)
    return boxes,scores

if __name__ == "__main__":
    tsum = 0
    idx = 0

    
    Path_Img = './test_img_carplate' 
    Path_pb = './data_pb_my/20180711_0p25_decay_0p0001_0p8.pb/frozen_inference_graph.pb'
    Path_Img_save = './result_v1/'

    #path = sys.argv[1]
    files = file_name(Path_Img)

    device ="/GPU:2"    

   #sess = tf.Session(config=config)
    sess = tf.Session()
    graph = sess.graph
    graph.as_default()

    with open(Path_pb,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

        with tf.device(device):
            image_tensor = graph.get_tensor_by_name('image_tensor:0')
            boxes =  graph.get_tensor_by_name('detection_boxes:0')
            scores = graph.get_tensor_by_name('detection_scores:0')
            classes = graph.get_tensor_by_name('detection_classes:0')
            num_detections = graph.get_tensor_by_name('num_detections:0')
    bad = 0
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    for f in files:
        print('===============================', idx, f)
        idx += 1
        if idx == 100:
            break
        fpath = os.path.join(Path_Img, f)
        box,score = run(fpath,sess,image_tensor,boxes,scores,classes,num_detections)
        print(score)
        
        image = cv2.imread(fpath)

        h,w,c = img.shape
        for i in range(len(score[0])):
            score_i = score[0][i]
            if score_i < 0.1:
                continue

            r=box[0][i]#产生不止一个值
            print(r)
            x1 = int(w*r[1])
            y1 = int(h*r[0])
            x2 = int(w*r[3])
            y2 = int(h*r[2])
            #img = img[y1:y2,x1:x2]

            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3) 
            cv2.putText(img,str(score[0][i])[0:4],(x1+5,y1+5),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255))
        dst = os.path.join(Path_Img_save, f)
        cv2.imwrite(dst, img)





import numpy as np
import cv2
import os
import sys
import tensorflow as tf
import time
from PIL import Image

def file_name(path):
    for root,dirs,files in os.walk(path):
        return files

def run(filename,sess,image_tensor,boxes,scores,classes,num_detections):
    image = Image.open(filename)
    image_np = np.array(image).astype(np.uint8)
    image_exp = np.expand_dims(image_np, axis=0)
    t1 = time.time() 
    (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_exp})
    print(scores)
    print(classes)
    print(boxes)
    return boxes,scores
def nms(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[0][:, 0]
    start_y = boxes[0][:, 1]
    end_x = boxes[0][:, 2]
    end_y = boxes[0][:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score[0])

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[0][index])
        picked_score.append(confidence_score[0][index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]
 
    return picked_boxes, picked_score

if __name__ == "__main__":
    tsum = 0
    idx = 0

    
    Path_Img = './test_img_carplate' 
    Path_pb = './data_pb_my/20180711_0p25_decay_0p0001_0p8.pb/frozen_inference_graph.pb'
    Path_Img_save = './result_v1/'

    #path = sys.argv[1]
    files = file_name(Path_Img)

    device ="/GPU:2"    

   #sess = tf.Session(config=config)
    sess = tf.Session()
    graph = sess.graph
    graph.as_default()

    with open(Path_pb,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

        with tf.device(device):
            image_tensor = graph.get_tensor_by_name('image_tensor:0')
            boxes =  graph.get_tensor_by_name('detection_boxes:0')
            scores = graph.get_tensor_by_name('detection_scores:0')
            classes = graph.get_tensor_by_name('detection_classes:0')
            num_detections = graph.get_tensor_by_name('num_detections:0')
    bad = 0

    threshold = 0.08
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    for f in files:
        print('===============================', idx, f)
        idx += 1
        if idx == 100:
            break
        fpath = os.path.join(Path_Img, f)
        box,score = run(fpath,sess,image_tensor,boxes,scores,classes,num_detections)
        print(score)
        
        image = cv2.imread(fpath)
     
        h,w,c = image.shape
        for  i in range(len(score[0])):
            r=box[0][i]
            x1 = int(w*r[1])
            y1 = int(h*r[0])
            x2 = int(w*r[3])
            y2 = int(h*r[2])
            box[0][i]=[x1,y1,x2,y2]
        print(box)

        picked_boxes, picked_score = nms(box, score, threshold)
        for (start_x, start_y, end_x, end_y), confidence in zip(picked_boxes, picked_score):
            print(confidence)
            (w, h), baseline = cv2.getTextSize(str(confidence), font, font_scale, thickness)
            print(w,h)
            cv2.rectangle(image, (int(start_x), int(start_y) - (2 * baseline + 5)), (int(start_x + w), int(start_y)), (0, 255, 255), -1)
            cv2.rectangle(image, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0, 255, 255), 2)
            cv2.putText(image, str(confidence), (int(start_x), int(start_y)), font, font_scale, (0, 0, 0), thickness)
        dst = os.path.join(Path_Img_save, f)
        cv2.imwrite(dst, image)
