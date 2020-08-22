import cv2
import numpy as np

"""
影片或攝影機讀取
"""
p1 = cv2.VideoCapture("video/1.mp4")
print("高:", p1.get(4))
print("寬:", p1.get(3))
print("總影格:", p1.get(7))
print("FPS:", p1.get(5))
p1.set(1, 500) # set 當前影格
while p1.isOpened()==True:
	ret, m1 = p1.read()
	if ret == True:
		print("當前影格:", p1.get(1))
		cv2.imshow("Image 1", m1)
		if cv2.waitKey(33) != -1: #按任意鍵離開(若沒按鍵則 cv2.waitKey() 會回傳-1)
			break
	else:
		break
p1.release()		
cv2.destroyAllWindows()



