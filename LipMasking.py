########################################################
#
#           20.08.12-13 by. JaeLin Joo
#
########################################################

# 입술 마스킹

import cv2
import numpy as np
import dlib

from PIL import Image
from PIL import ImageFilter

# 이미지 불러오기
img = cv2.imread("./source/ha2.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
lip_mask = np.zeros_like(img_gray)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_194_face_landmarks.dat")
faces = detector(img_gray)
for face in faces:
    landmarks = predictor(img_gray, face)
    landmarks_points = []
    
    #입술
    for n in range(11,21):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))
    for n in range(22,26):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))
    for n in range(152,194):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))
    
    points = np.array(landmarks_points, np.int32)
    
    convexhull = []
    ##### 아랫입술 ###### 6
    convexhull.append(cv2.convexHull(points[[0,27,28]]))
    convexhull.append(cv2.convexHull(points[[0,1,2,3,28,29,30,31]]))
    convexhull.append(cv2.convexHull(points[[3,4,5,6,31,32,33,34]]))
    convexhull.append(cv2.convexHull(points[[6,7,8,9,34,35,36]]))
    convexhull.append(cv2.convexHull(points[[9,10,11,12,36,37,38,39]]))
    convexhull.append(cv2.convexHull(points[[12,13,39,40,41]]))
    ##### 윗입술 ##### 10
    convexhull.append(cv2.convexHull(points[[14,15,42,43]]))
    convexhull.append(cv2.convexHull(points[[15,16,17,43,44,45]]))
    convexhull.append(cv2.convexHull(points[[17,18,45,46]]))
    convexhull.append(cv2.convexHull(points[[18,19,46,47]]))
    convexhull.append(cv2.convexHull(points[[19,20,47,48]]))
    convexhull.append(cv2.convexHull(points[[20,21,48,49]]))
    convexhull.append(cv2.convexHull(points[[21,22,49,50,51]]))
    convexhull.append(cv2.convexHull(points[[22,23,51,52]]))
    convexhull.append(cv2.convexHull(points[[23,24,52,53,54]]))
    convexhull.append(cv2.convexHull(points[[24,25,26,54,55]]))
    ## 중간 ##
    convexhull.append(cv2.convexHull(points[[26,27,55]]))
    convexhull.append(cv2.convexHull(points[[14,41,42]]))
    convexhull.append(cv2.convexHull(points[[13,41,42]]))
    
    for con in convexhull:
        cv2.polylines(img, [con], True, (255, 0, 0), 1)
        cv2.fillConvexPoly(lip_mask, con, 255)
    

    lip_image = cv2.bitwise_and(img, img, mask=lip_mask)
    

# file 저장
cv2.imwrite("./mask/lip_mask.PNG", lip_mask)

# masking한 부분만 놓고 나머지 배경 투명하게 처리

MAKE_TRANSPARENT = True

if(MAKE_TRANSPARENT):
    img = Image.open("./mask/lip_mask.PNG") # 파일 열기
    img = img.convert("RGBA") #RGBA형식으로 변환 
    
    datas = img.getdata() #datas에 일차원 배열 형식으로 RGBA입력
    newData = []

    for item in datas:
     
        if (item[0]==255 and item[1]== 255 and item[2] == 255): #해당 픽셀 색이 흰색이면 해당 영역 추가
            newData.append((200,0,10,100)) 
        else: #그렇지 않으면
            newData.append((255,0,0, 0))  # 투명 추가


    img.putdata(newData) #데이터 입력
    img = img.filter(ImageFilter.GaussianBlur(radius=6))
    img.save("./mask/lip_mask.PNG") 
    




