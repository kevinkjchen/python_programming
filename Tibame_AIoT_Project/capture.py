# 從影片擷取圖片: 從影片中每隔幾秒秀一張圖, 再按一個鍵決定這張圖是否要存成圖片檔
import os
import glob
import cv2
import numpy as np

path = input("輸入目錄名:") 
period = float(input("輸入每幾秒抓一張圖:"))
base = input("輸入base檔名 [0]:0000, [1]:1000, [2]:2000, [3]:3000...:")

files=glob.glob(path+'/*.mp4')
print(files)
i = 0
for fn in files:
	p1 = cv2.VideoCapture(fn)
	print("檔名:", fn)
	print("總影格:", p1.get(7))
	fps = p1.get(5)

	#p1.set(1, 5000)

	while p1.isOpened()==True:
		ret, m1 = p1.read()
		if ret == True:
			#print("當前影格:", p1.get(1))
			cv2.imshow("Image 1", m1)
			
			key = cv2.waitKey(0)
			if key == 115: #'s'
				cv2.imwrite('capture/img_{}{:03d}.jpg'.format(base, i), m1, [cv2.IMWRITE_JPEG_QUALITY, 100])
				i += 1
			elif key == 101: #'e'
				break
				
			#print(key)	
		else:
			break
			
		# 每幾秒抓一張圖	
		p1.set(1, p1.get(1) + fps * period)	
		
	p1.release()		
	cv2.destroyAllWindows()




