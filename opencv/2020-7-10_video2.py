
import cv2
import numpy as np
# 讀影片 處理 另存新檔

# 取得影像來源, 建立控制變數
p1 = cv2.VideoCapture("video/1.mp4")
#p1 = cv2.VideoCapture(0)
f = cv2.VideoWriter_fourcc(*'MP4V')
w = int(p1.get(3))
h = int(p1.get(4))
# 建立儲存控制變數
p2 = cv2.VideoWriter("video/2.mp4", f, 30, (w,h))
while p1.isOpened()==True:
	ret, m1 = p1.read()
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



