# -*- coding: utf-8 -*-
import os
import random
import cv2
import sys
import numpy as np
import tensorflow as tf
import time
from datetime import datetime as dt
import shutil
import serial

ser = serial.Serial('/dev/ttyUSB0', 19200)
DEVICE_ID = 0
path = "./faces/train"
dirs = os.listdir(path)
dirs = [f for f in dirs if os.path.isdir(os.path.join(path, f))]
label_dict = {}
i = 0
faces = []
for dirname in dirs:
    label_dict[dirname] = i
    i += 1

names = dirs

def get_batch_list(l, batch_size):
    # [1, 2, 3, 4, 5,...] -> [[1, 2, 3], [4, 5,..]]
    return [np.asarray(l[_:_+batch_size]) for _ in range(0, len(l), batch_size)]

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def inference(images_placeholder, keep_prob):

    x_image = tf.reshape(images_placeholder, [-1, 32, 32, 3])

    # Convolution layer
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer
    h_pool1 = max_pool_2x2(h_conv1)

    # Convolution layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Pooling layer
    h_pool2 = max_pool_2x2(h_conv2)

    # Full connected layer
    W_fc1 = weight_variable([8 * 8 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Full connected layer
    W_fc2 = weight_variable([1024, len(label_dict)])
    b_fc2 = bias_variable([len(label_dict)])

    return tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# cv2.cv.CV_FOURCC
def cv_fourcc(c1, c2, c3, c4):
    return (ord(c1) & 255) + ((ord(c2) & 255) << 8) + \
        ((ord(c3) & 255) << 16) + ((ord(c4) & 255) << 24)

if __name__ == '__main__':

    # 定数定義
    ESC_KEY = 27     # Escキー
    INTERVAL= 33     # 待ち時間
    FRAME_RATE = 120  # fps

    WINDOW_NAME = "detect"
    #FILE_NAME = "detect.avi"

    # 分類器の指定
    cascade_file = "cascades/haarcascade_frontalface_alt2.xml"
    cascade = cv2.CascadeClassifier(cascade_file)

    # カメラ映像取得
    cap = cv2.VideoCapture(DEVICE_ID)

    end_flag, c_frame = cap.read()
    height, width, channels = c_frame.shape

    # 保存ビデオファイルの準備
    #rec = cv2.VideoWriter(FILE_NAME, cv_fourcc('X', 'V', 'I', 'D'), FRAME_RATE, (width, height), True)

    # ウィンドウの準備
    cv2.namedWindow(WINDOW_NAME)

    font = cv2.FONT_HERSHEY_PLAIN
    font_size = 1.5

    print("setup tensorflow")
    x = tf.placeholder('float', shape=[None, 32 * 32 * 3])  # 32 * 32, 3 channels
    keep_prob = tf.placeholder('float')
    y_conv = inference(x, keep_prob)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    print("loading model data")
    tf.train.Saver().restore(sess, "model_face/model.ckpt")

    # 変換処理ループ
    while end_flag == True:

        img = c_frame

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img_gray = img
        face_list = cascade.detectMultiScale(img_gray, minSize=(150, 150))

        for (pos_x, pos_y, w, h) in face_list:

            img_face = img[pos_y:pos_y+h, pos_x:pos_x+w]

            img_face1 = cv2.resize(img_face, (32, 32))
            img_face2 = cv2.resize(img_face, (100, 100))
            faces.append(img_face2)
            test_images = []
            test_images.append(img_face1.flatten().astype(np.float32) / 255.0)
            test_images = np.asarray(test_images)

            results = y_conv.eval(feed_dict={ x: test_images, keep_prob: 1.0 })
            text = names[np.argmax(results[0])]
            
            color = (0, 0, 225)
            pen_w = 2
            cv2.putText(img,text,(pos_x,pos_y - 10),font,font_size,(255,255,0))
            cv2.rectangle(img, (pos_x, pos_y), (pos_x+w, pos_y+h), color, thickness = pen_w)
            cv2.imshow(WINDOW_NAME, img)
            print("Hello,", text)
        # フレーム表示
            face = cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
            tdatetime = dt.now()
            tstr = tdatetime.strftime('%Y%m%d%H%M%S')
            current_dir = os.getcwd()
            O='o\r\n'
            OE=O.encode('utf-8')
            if len(face) > 0:
                ser.write(OE)
                check = 'here/'
                ath = '{}/faces/train/{}'.format(os.getcwd(), text)
                os.makedirs(here, exist_ok=True)
                os.chdir(here)
                os.rmdir('here/{}'.format(text))
                os.chdir(ath)
                #os.makedirs(text, exist_ok=True)
                #os.chdir(text)
                #cv2.imwrite("detected.jpg", img)
                cv2.imwrite(tstr+'.jpg', img_face2)
                os.chdir('../')
                print("Detected", tstr)
                time.sleep(20)

            if len(face) == 0:
                print("Could'nt find")
        # 次のフレーム読み込み
        end_flag, c_frame = cap.read()

    # 終了処理
    sess.close()

    cv2.destroyAllWindows()
    print(len(faces))
    cap.release()
    #rec.release()