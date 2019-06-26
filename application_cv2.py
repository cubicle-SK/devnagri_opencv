
import numpy as np
import cv2
from keras.models import load_model
from collections import deque #queue

model= load_model('devnagri.h5')
print(model)

letters={0:'CHECK',1:'character_1_ka',2:'character_2_kha',
         3:'character_3_ga',4:'character_4_gha',5:'character_5_kna',
         6:'character_06_cha',7:'character_7_chha',8:'character_8_ja',
         9:'character_9_jha',10:'character_10_yna',
         11:'character_11_taamatar',12:'character_12_thaa',
         13:'character_13_daa',14:'charater_14_dhaa',
         15:'character_15_adna',16:'character_16_tabala',
         17:'character_17_tha',18:'character_18_da',19:'character_19_dha',
         20:'character_20_na',21:'character_21_pa',22:'character_22_pha',
         23:'character_23_ba',24:'character_24_bha',25:'character_25_ma',
         26:'character_26_yaw',27:'character_27_ra',28:'character_28_la',
         29:'character_29_waw',30:'character_30_motosaw',31:'character_31_petchiryakha',
         32:'character_32_patalosaw',33:'character_33_ha',34:'character_34_chhya',
         35:'character_35_tra',36:'character_36_gya',
         37:'CHECK'}

def k_pred(model, image):
    process=k_pro_im(image)
    print('processed:', str(process.shape))
    pred_p= model.predict(process)[0]
    pred_class=list(pred_p).index(max(pred_p))
    return max(pred_p), pred_class

def k_pro_im(img):
    image_x=32
    image_y=32
    img=cv2.resize(img,(image_x,image_y))
    img=np.array(img, dtype=np.float32)
    img=np.reshape(img,(-1,image_x, image_y,1))
    print('img:',img.shape)
    return img

k_pred(model, np.zeros((32, 32, 1), dtype=np.uint8))
cap=cv2.VideoCapture(0)
l_blue=np.array([110,50,50])
u_blue=np.array([130,255,255])
pred_class=0
ptx=deque(maxlen=512)
blackboard=np.zeros((480,640,3), dtype=np.uint8)
digit= np.zeros((200,200,3), dtype=np.uint8)
while(cap.isOpened()):
  ret,img=cap.read()
  img=cv2.flip(img,1)
  imgHSV=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  mask=cv2.inRange(imgHSV,l_blue,u_blue)
  blur=cv2.medianBlur(mask,15) 
  blur=cv2.GaussianBlur(blur,(5,5),0)
  thresh=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
  cnts=cv2.findContours(thresh.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
  center=None
  if len(cnts)>=1:
    contour= max(cnts, key=cv2.contourArea)
    if cv2.contourArea(contour)>250:
      ((x,y),rad)=cv2.minEnclosingCircle(contour)
      cv2.circle(img,(int(x),int(y)), int(rad),(0,255,255),2)
      cv2.circle(img, center,5,(0,0,255),-1)
      M=cv2.moments(contour)
      center=(int(M['m10']/M['m00']), int(M['m01']/M['m00']))
      ptx.appendleft(center)
      for i in range(1, len(ptx)):
        if ptx[i-1] is None or ptx[i] is None:
          continue
        cv2.line(blackboard,ptx[i-1],ptx[i], (255,255,255),10)
        cv2.line(img,ptx[i-1], ptx[i], (0,0,255),5)
  elif len(cnts)==0:
    if len(ptx) !=[]:
      blackboard_gray= cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
      blur1= cv2.medianBlur(blackboard_gray,15)
      blur1= cv2.GaussianBlur(blur1,(5,5),0)
      thresh1= cv2.threshold(blur1,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
      blackboard_cnts=cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
      if len(blackboard_cnts) >=1:
        cnt=max(blackboard_cnts, key=cv2.contourArea)
        print(cv2.contourArea(cnt))
        if cv2.contourArea(cnt)>2000:
          x,y,w,h=cv2.boundingRect(cnt)
          digit=blackboard_gray[y:y+h,x:x+w]
          print("digit:",digit.shape)
          pred_p,pred_class= k_pred(model,digit)
          print(pred_class,pred_p)
    ptx=deque(maxlen=512)
    blackboard=np.zeros((480,640,3), dtype=np.uint8)
  cv2.putText(img,'conv network:'+str(letters[pred_class]),(800,470),
             cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
  cv2.imshow('frame', img)
  cv2.imshow('contours', thresh)
  k=cv2.waitKey(10)
  if k==27:
    break


