''''
Capture multiple Faces from multiple users to be stored on a DataBase (dataset directory)
	==> Faces will be stored on a directory: dataset/ (if does not exist, pls create one)
	==> Each face will have a unique numeric integer ID as 1, 2, 3, etc                       

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18    

'''

import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

while(True):

    ret, img = cam.read()
    #  图像水平垂直翻转
    img = cv2.flip(img, -1) # flip video image vertically
    # 将图像转换为灰度图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #1.image表示的是要检测的输入图像
    #2.objects表示检测到的人脸目标序列
    # 参数scaleFactor：是尺度变换，就是向上或者向下每次是原来的多少倍，这里是1.3倍
    #3.scaleFactor表示每次图像尺寸减小的比例
    #4. minNeighbors表示每一个目标至少要被检测到3次才算是真的目标(因为周围的像素和不同的窗口大小都可以检测到人脸),
    #5.minSize为目标的最小尺寸
    #6.minSize为目标的最大尺寸
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        # 绘画矩形
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)
    #等待 键盘输入
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    # 输入 ESC  跳出数据录入
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()


