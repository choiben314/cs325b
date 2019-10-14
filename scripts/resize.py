from PIL import Image
import os, sys

img_path = '/home/BenChoi/cs325b/roadtype/kenya_road_images/'
save_path = '/home/BenChoi/cs325b/local_images/kenya/'

old_dirs = os.listdir(img_path)
dirs = old_dirs[::-1] # reverses

print("-----------------RESIZING KENYA------------------")

def resize(country, num):
	i = 0
	for item in dirs:
		if os.path.isfile(img_path + item):
			try:
				im = Image.open(img_path+item)
			except:
				print(item + " is corrupt.")
				error_file = open("errors.txt", "a")
				error_file.write(item + '\n') 
				error_file.close()
				continue
			f = item.split('_')[-1][:-4]
			imResize = im.resize((224, 224))
			imResize = imResize.convert("RGB")
			imResize.save(save_path + f + '_' + country + '_resized.jpg', "JPEG", quality=90)
		if i > num:
			break
		i += 1

resize('kenya', 104000)

img_path = '/home/BenChoi/cs325b/roadtype/peru_road_images/'
save_path = '/home/BenChoi/cs325b/local_images/peru/'

dirs = os.listdir(img_path)

print("-----------------RESIZING PERU------------------")

resize ('peru', 500000)
