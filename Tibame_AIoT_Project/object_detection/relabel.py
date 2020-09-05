# 找出yolos資料夾下有成對的img_xxxx.txt與img_xxxx.jpg,如果有標註person類別,
# 把txt檔的類別16改為類別0,另存檔至yolos_person資料夾

import os
import shutil
from glob import glob

#src_path = input("輸入原始路徑:") + '/'
#des_path = input("輸入目標路徑:") + '/'
src_path = 'yolos/'
des_path = 'yolos_person/'

files = glob(src_path+'img_*.txt')
for file in files:
	#fn = file.split('.')[0].split('/')[-1]	# for linux
	fn = file.split('.')[0].split('\\')[-1]	# for windows
	#print(file, fn)
	if os.path.isfile(src_path+fn+'.jpg'):
		with open(file, 'r', encoding="utf-8") as f:
			lines = f.readlines()
			#把 img_xxxx.txt 檔的類別 16 改為類別 0, 另存檔至 yolos_person/
			new_file_lines = []
			count = 0
			for line in lines:
				if line.split(' ')[0] == '16':
					new_file_lines.append(line.replace('16', '0'))
					count += 1
			if count != 0:
				shutil.copyfile(file.split('.')[0]+'.jpg', des_path+fn+'.jpg')
				print("copy", file.split('.')[0]+'.jpg', "to", des_path+fn+'.jpg')
				with open (des_path+fn+'.txt', 'a', encoding="utf-8") as f1:
					for ln in new_file_lines:
						f1.write(ln)
			else:
				print(fn+'.txt has no person class')

	else:
		 print(fn+'.jpg', "not exist")
			
			
