import sys
import dlib
import cv2
from PIL import Image
import numpy as np
import time
import os
import base64


def get_base64(path):
    with open(path, 'rb')as f:
        b64 = str(base64.b64encode(f.read()), 'utf-8')
    return b64


request_url = "https://aip.baidubce.com/rest/2.0/face/v3/detect"

from aip import AipFace

# 你的appId, apiKey, secretKey
af = AipFace('', '', '')

dir = '表情包/'

for path in os.listdir(dir):
    try:
        if '.png' in path:
            png = Image.open(path).convert('RGB')
            png.save('temp.jpg')
            path = 'temp.jpg'
        json = af.detect(get_base64(dir + path), 'BASE64')
        if json['result'] == None:
            continue

        face_list = json['result']['face_list']

        for face_num, face in enumerate(face_list):
            face_img = Image.open(dir + path)

            face_width = face['location']['width']
            face_height = face['location']['height']
            # 脸距离顶部的距离
            face_top = int(face['location']['top'])
            # 脸距离左部的距离
            face_left = int(face['location']['left'])
            face_right = int(face_left + face_width)
            face_bottom = face_top + face_height
            hat = Image.open('hat2.jpg')
            # 调整帽子大小
            face_width = int((face_right - face_left) * 1.35)
            hat_x_offset = int(0.2 * face_width)
            hat_y_offset = int(0.9 * face_height)
            hat = hat.resize((face_width, face_height))
            b1 = face_left - hat_x_offset
            a1 = face_top - hat_y_offset
            # rotated_b1=int(np.sin(np.pi*rotation/180)*face_width-np.cos(np.pi*rotation/180)*face_height)
            # rotated_a1=int(np.cos(np.pi*rotation/180)*face_width+np.sin(np.pi*rotation/180)*face_height)

            box=(b1,a1,face_left+face_width-hat_x_offset,face_top+face_height-hat_y_offset)

            # box = (b1,a1,face_left+face_width-hat_x_offset,face_top+face_height-hat_y_offset)


            arr_hat = np.array(hat)
            # 转化为灰度图
            hat_gray = np.array(hat.convert('L'))

            # crop剪出一个矩形区域
            # box（b1,a1,b2,a2）
            print(box,face_width,face_height)
            arr_face = np.array(face_img.crop(box))
            # 二值化
            ret, mask = cv2.threshold(hat_gray, 247, 255, cv2.THRESH_BINARY)

            # 取反
            mask_inv = cv2.bitwise_not(mask)

            # 跟hat图片与操作
            hat_bg = cv2.bitwise_and(arr_face, arr_face, mask=mask)

            # 与操作
            face_fg = cv2.bitwise_and(arr_hat, arr_hat, mask=mask_inv)

            # 通过array创建新图片
            result = Image.fromarray(cv2.add(face_fg, hat_bg))
            face_img.paste(result, box)
            name = face['face_token']
            # face_img.show()
            face_img.save(r'戴帽子后\%s.jpg' % name)
        time.sleep(0.1)
    except:
        pass
