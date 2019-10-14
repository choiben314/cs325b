from PIL import Image
import os, sys

path = '/home/BenChoi/cs325b/test_images/kenya_road_images/'
dirs = os.listdir(path)

def resize():
	i = 0
	for item in dirs:
		if os.path.isfile(path+item):
			im = Image.open(path+item)
			x, y = im.size
			print(x, y)
			f, e = os.path.splitext(path+item)
			imResize = im.resize((224, 224))
			imResize.save(f + '_resized.tif')

resize()
