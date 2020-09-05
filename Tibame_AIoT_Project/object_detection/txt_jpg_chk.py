# For YOLOv4, 檢查訓練資料是否有成對的jpg和txt檔 
import os
import shutil
from glob import glob

src_path = input("輸入原始路徑:") + '/'
des_path = input("輸入目標路徑:") + '/'
#src_path = 'yolos/'
#des_path = 'yolos_person/'

txt_files = glob(src_path+'img_*.txt')
for file in txt_files:
	
	#fn = file.split('.')[0].split('/')[-1]	# for linux
	fn = file.split('.')[0].split('\\')[-1]	# for windows
	print(file, fn)
	if os.path.isfile(src_path+fn+'.jpg'):
		pass
	else:
		 print(fn+'.jpg', "not exist")
				
			
jpg_files = glob(src_path+'img_*.jpg')
for file in jpg_files:
	#fn = file.split('.')[0].split('/')[-1]	# for linux
	fn = file.split('.')[0].split('\\')[-1]	# for windows
	print(file, fn)
	if os.path.isfile(src_path+fn+'.txt'):
		pass
	else:
		 print(fn+'.txt', "not exist")	