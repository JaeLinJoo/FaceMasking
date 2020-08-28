#!/usr/bin/env python
# coding: utf-8

# In[47]:


########################################################
#
#           20.08.18  by. JaeLin Joo
#
########################################################

# 눈썹 마스킹

import cv2
import numpy as np
import dlib
from scipy.spatial import distance
from PIL import Image
from PIL import ImageFilter

# 이미지 불러오기
img = cv2.imread("./source/9.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eyebrow_mask = np.zeros_like(img_gray)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_194_face_landmarks.dat")
faces = detector(img_gray)
for face in faces:
    landmarks = predictor(img_gray, face)
    landmarks_points = []
    
    p101 = (landmarks.part(101).x, landmarks.part(101).y)
    p105 = (landmarks.part(105).x, landmarks.part(105).y)
    dst1 = distance.euclidean(p101,p105)
    
    p80 = (landmarks.part(80).x, landmarks.part(80).y)
    p82 = (landmarks.part(82).x, landmarks.part(82).y)
    dst2 = distance.euclidean(p80,p82)
    
    # left eyebrow
    for n in range(92,95):
        x = landmarks.part(n).x 
        y = landmarks.part(n).y 
        landmarks_points.append((x, y))
    for n in range(95,98):
        x = landmarks.part(n).x 
        y = landmarks.part(n).y + dst1*0.08
        landmarks_points.append((x, y))
    x = landmarks.part(99).x
    y = landmarks.part(99).y + dst1*0.15
    landmarks_points.append((x, y))
    x = landmarks.part(100).x
    y = landmarks.part(100).y + dst1*0.2
    landmarks_points.append((x, y))
    x = landmarks.part(101).x
    y = landmarks.part(101).y + dst1*0.25
    landmarks_points.append((x, y))
    for n in range(102,106):
        x = landmarks.part(n).x
        y = landmarks.part(n).y - dst1*0.13
        landmarks_points.append((x, y))
    x = landmarks.part(106).x
    y = landmarks.part(106).y - dst1*0.23
    landmarks_points.append((x, y))
    x = landmarks.part(107).x
    y = landmarks.part(107).y - dst1*0.15
    landmarks_points.append((x, y))
    x = landmarks.part(108).x
    y = landmarks.part(108).y - dst1*0.15
    landmarks_points.append((x, y))
    x = landmarks.part(110).x
    y = landmarks.part(110).y - dst1*0.2
    landmarks_points.append((x, y))
    x = landmarks.part(111).x
    y = landmarks.part(111).y - dst1*0.25
    landmarks_points.append((x, y))
    x = landmarks.part(112).x
    y = landmarks.part(112).y - dst1*0.28
    landmarks_points.append((x, y))
    x = landmarks.part(113).x
    y = landmarks.part(113).y - dst1*0.25
    landmarks_points.append((x, y))
    
    #right eyebrow
    for n in range(70,73):
        x = landmarks.part(n).x
        y = landmarks.part(n).y - dst2*0.05
        landmarks_points.append((x, y))
    for n in range(73,76):
        x = landmarks.part(n).x
        y = landmarks.part(n).y + dst2*0.05
        landmarks_points.append((x, y))
    x = landmarks.part(77).x
    y = landmarks.part(77).y + dst2*0.1
    landmarks_points.append((x, y))
    x = landmarks.part(78).x
    y = landmarks.part(78).y + dst2*0.2 
    landmarks_points.append((x, y))
    x = landmarks.part(79).x
    y = landmarks.part(79).y + dst2*0.25
    landmarks_points.append((x, y))
    for n in range(80,83):
        x = landmarks.part(n).x
        y = landmarks.part(n).y - dst2*0.13
        landmarks_points.append((x, y))
    x = landmarks.part(83).x
    y = landmarks.part(83).y - dst2*0.3
    landmarks_points.append((x, y)) 
    for n in range(84,87):
        x = landmarks.part(n).x
        y = landmarks.part(n).y - dst2*0.2
        landmarks_points.append((x, y))
    x = landmarks.part(88).x
    y = landmarks.part(88).y - dst2*0.25
    landmarks_points.append((x, y)) 
    x = landmarks.part(89).x
    y = landmarks.part(89).y - dst2*0.3
    landmarks_points.append((x, y)) 
    x = landmarks.part(90).x
    y = landmarks.part(90).y - dst2*0.3
    landmarks_points.append((x, y)) 
    x = landmarks.part(91).x
    y = landmarks.part(91).y - dst2*0.3
    landmarks_points.append((x, y)) 
    

    points = np.array(landmarks_points, np.int32)
    
    convexhull = []
    # left eyebrow
    convexhull.append(cv2.convexHull(points[[1,2,18,19]]))
    convexhull.append(cv2.convexHull(points[[2,3,17,18]]))
    convexhull.append(cv2.convexHull(points[[3,4,16,17]]))
    convexhull.append(cv2.convexHull(points[[4,5,15,16]]))
    convexhull.append(cv2.convexHull(points[[5,6,14,15]]))
    convexhull.append(cv2.convexHull(points[[6,7,13,14]]))
    convexhull.append(cv2.convexHull(points[[7,8,13]]))
    convexhull.append(cv2.convexHull(points[[8,10,13]]))
        
    #right eyebrow
    convexhull.append(cv2.convexHull(points[[21,22,38,39]]))
    convexhull.append(cv2.convexHull(points[[22,23,37,38]]))
    convexhull.append(cv2.convexHull(points[[23,24,36,37]]))
    convexhull.append(cv2.convexHull(points[[24,25,35,36]]))
    convexhull.append(cv2.convexHull(points[[25,26,34,35]]))
    convexhull.append(cv2.convexHull(points[[26,27,33,34]]))
    convexhull.append(cv2.convexHull(points[[27,28,33]]))
    convexhull.append(cv2.convexHull(points[[28,30,33]]))
        
    
    for con in convexhull:
        cv2.polylines(img, [con], True, (255, 0, 0), 1)
        cv2.fillConvexPoly(eyebrow_mask, con, 255)
    

    eyebrow_image_1 = cv2.bitwise_and(img, img, mask=eyebrow_mask)

# file 저장
cv2.imwrite("./mask/eyebrow_mask.PNG", eyebrow_mask)


# masking한 부분만 놓고 나머지 배경 투명하게 처리

MAKE_TRANSPARENT = True

if(MAKE_TRANSPARENT):
    img = Image.open("./mask/eyebrow_mask.PNG") # 파일 열기
    img = img.convert("RGBA") #RGBA형식으로 변환 
    
    datas = img.getdata() #datas에 일차원 배열 형식으로 RGBA입력
    newData = []

    for item in datas:
     
        if (item[0]==255 and item[1]== 255 and item[2] == 255): #해당 픽셀 색이 흰색이면 해당 영역 추가
            newData.append((0,0,0,50)) 
        else: #그렇지 않으면
            newData.append((0,0,0, 0))  # 투명 추가


    img.putdata(newData) #데이터 입력
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    img.save("./mask/eyebrow_mask2.PNG") 


# In[ ]:




