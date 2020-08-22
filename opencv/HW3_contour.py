import cv2
import numpy as np

p1 = cv2.VideoCapture("video/homework3.mp4")
# print("高:", p1.get(4))
# print("寬:", p1.get(3))
# print("總影格:", p1.get(7))
# print("FPS:", p1.get(5))
while p1.isOpened()==True:
	ret, m1 = p1.read()
	if ret == True:
		#print("當前影格:", p1.get(1))
		#m2 = cv2.inRange(m1, (100,30,20),(200,80,55))
		m2 = cv2.inRange(m1, (100,20,10), (200,100,90))
		#m2 = cv2.dilate(m2, (45,45))	
		#m2 = cv2.dilate(m2, (100,100))	
		m2 = cv2.morphologyEx(m2, cv2.MORPH_CLOSE, np.ones((50,50)))
		#m2 = cv2.morphologyEx(m2, cv2.MORPH_GRADIENT, np.ones((50,50)))

		#cv2.imwrite("output/pen.jpg", m1)
		#cv2.imwrite("output/pen2.jpg", m2)
		#cv2.imshow("Image 2", m2)
		a,b = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		#print(len(a))
		if len(a)>0:
			x,y,w,h = cv2.boundingRect(a[0])
			cv2.rectangle(m1,(x,y),(x+w,y+h),(0,0,255),2)
		
		cv2.imshow("Image 1", m1)


		if cv2.waitKey(33) != -1: #按任意鍵離開(若沒按鍵則 cv2.waitKey() 會回傳-1)
			break
	else:
		break
p1.release()		
cv2.destroyAllWindows()