import os
import sys
import time
from datetime import datetime as dt
import cv2

name = sys.argv[-1]
print(name)

cascade_file = "cascades/haarcascade_frontalface_alt2.xml"
cascade = cv2.CascadeClassifier(cascade_file)

cnt_max = 11
cnt_face = 0
faces = []

# カメラ取り込み開始
DEVICE_ID = 0
cap = cv2.VideoCapture(DEVICE_ID)

# 初期フレームの取得
time.sleep(1)
end_flag, c_frame = cap.read()
height, width, channels = c_frame.shape

while cnt_face < cnt_max:

    img = c_frame

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_list = cascade.detectMultiScale(img_gray, minSize=(100, 100))

    for (pos_x, pos_y, w, h) in face_list:
        img_face = img[pos_y:pos_y+h, pos_x:pos_x+w]
        # 画像サイズ変更
        img_face = cv2.resize(img_face, (100, 100))
        faces.append(img_face)

    if len(face_list) > 0:
        cnt_face += 1
        print("\r", cnt_face, end="")

    time.sleep(1)
    end_flag, c_frame = cap.read()

cap.release()

print(len(faces))

num = 0
tdatetime = dt.now()
tstr = tdatetime.strftime('%Y%m%d%H%M')

# 学習データ
path = '{}/faces/train/{}'.format(os.getcwd(), name)
print("学習データ", path)
os.makedirs(path)

for face in faces[:-1]:
    filename = '{}-{}.jpg'.format(tstr, num)
    print("\t", filename)
    cv2.imwrite('{}/{}'.format(path, filename), face)
    num += 1

# テストデータ
path = '{}/faces/test/{}'.format(os.getcwd(), name)
print("テストデータ", path)
os.makedirs(path)

face = faces[-1]
filename = '{}-{}.jpg'.format(tstr, num)
print("\t", filename)
cv2.imwrite('{}/{}'.format(path, filename), face)