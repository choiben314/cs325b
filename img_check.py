import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

#x = os.listdir('data/kenya/kenya_1000x1000_images')
x = os.listdir('data/peru/224/cropped')

corrupt = open('data/peru/corrupt.txt', 'w')
cloud = open('data/peru/cloudy.txt', 'w')
gray = open('data/peru/grayscale.txt', 'w')

for idx, el in enumerate(x):
    if idx % 1000 == 0:
        print(idx)
    img = cv2.imread('data/peru/224/cropped/' + el, cv2.IMREAD_COLOR)
    if img is None:
        corrupt.write(el + '\n')
    else:
        rgb_mean = list(cv2.mean(img))[:-1]
        if all(i > 150 for i in rgb_mean):
            cloud.write(el + '\n')
        elif rgb_mean.count(rgb_mean[0]) == len(rgb_mean):
            gray.write(el + '\n')

corrupt.close()
cloud.close()
gray.close()
