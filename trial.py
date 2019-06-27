import numpy as np
from keras.models import load_model
#from keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
model= load_model('devnagri.h5')

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
    print('list:',list(pred_p))
    pred_class=list(pred_p).index(max(pred_p))
    return max(pred_p), pred_class+1

def k_pro_im(img):
    img=cv2.resize(img,(32,32))
    img=np.array(img, dtype=np.float32)
    img=np.reshape(img,(-1,32,32,1))
    print('img:',img.shape)
    return img

digit =cv2.imread('/Users/saumyakansal/Desktop/SAUMYA/Python and ML/Dev/Test/character_35_tra/3547.png',
                  cv2.IMREAD_UNCHANGED)
print(digit.shape)
plt.imshow(digit)
plt.show()
pred_p,pred_class= k_pred(model,digit)
print('class:',pred_class)
print("The class of the input Image : ",letters[pred_class])