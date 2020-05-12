    # Ref: https://github.com/jagracar/OpenCV-python-tests/blob/master/OpenCV-tutorials/featureDetection/fast.py
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('predict.png',0)

# iniciando o FAST com valores default
fast = cv2.FastFeatureDetector_create(threshold=20)

# encontrar e desenhar os pontos chaves
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp, None,color=(255,0,0))

print("Threshold: ", fast.getThreshold())
print("nonmaxSuppression: ", fast.getNonmaxSuppression())
print("neighborhood: ", fast.getType())
print("Total Keypoints with nonmaxSuppression: ", len(kp))

cv2.imwrite('fast_true20.png',img2)

# Desabilitando a nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img,None)

print ("Numero total de pontos chave sem nonmaxSuppression: ", len(kp))

img3 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))

cv2.imwrite('fast_false20.png',img3)