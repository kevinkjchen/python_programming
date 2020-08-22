import os
import cv2
import numpy as np

# 取得影像來源, 建立控制變數
fn="video/1.mp4"
p1 = cv2.VideoCapture(fn)
w1 = int(p1.get(3))
h1 = int(p1.get(4))
fps = p1.get(5)
print("檔名:", fn)
print("總影格:", p1.get(7))
print("每秒的影格數:", fps)
print("w=", w1, "h=", h1)

#p1.set(1, 1000)

f = cv2.VideoWriter_fourcc(*'MP4V')
# 建立儲存控制變數
p2 = cv2.VideoWriter("2.mp4", f, 30, (w1,h1))
while p1.isOpened()==True:
	# if p1.get(1) >　last_frame:
	# 	break
	ret, m1 = p1.read()
	#print("影格:", p1.get(1))

	# # 圖像縮放
	# w2 = w1//2
	# h2 = int(w2/m1.shape[1]*m1.shape[0])
	# m2 = cv2.resize(m1, (w2,h2))	
	# cv2.imshow("Image 1", m2)	#顯示圖像
	# cv2.waitKey(0)
	# p1.set(1, p1.get(1) + fps * period)		

	if ret == True:
		p2.write(m1)	#寫入圖像
		# 如果是從攝影機擷取, 以下三行不可省略
		# cv2.imshow("Image 1", m1)	#顯示圖像
		# if cv2.waitKey(33) != -1: #按任意鍵離開(若沒按鍵則 cv2.waitKey() 會回傳-1)
		# 	break
	else:
		break
# 釋放控制變數
p1.release()		
cv2.destroyAllWindows()