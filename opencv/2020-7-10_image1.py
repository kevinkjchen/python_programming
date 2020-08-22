
import cv2
import numpy as np

# 繪圖
m1 = np.full(
	(150, 300, 3), 		# 設定每個維度的長度
	(200, 200, 255), 	# color初始值
	np.uint8			# 陣列型態
)
cv2.line(m1, (20,10), (100, 10), (255,255,255), 2)
cv2.rectangle(m1, (20,15), (100, 80), (255,255,255), 2)
cv2.rectangle(m1, (20,100), (100, 140), (255,255,255), -1)
cv2.circle(m1, (120, 75), 50, (255,0,0), 2)
cv2.circle(m1, (220, 75), 50, (255,0,0), -1)

cv2.imshow("Image 1", m1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 如果要載入自訂的字型，需使用PIL函式庫的Image
from PIL import ImageFont, ImageDraw, Image

m1 = np.full(
	(150, 300, 3), 		# 設定每個維度的長度
	(200, 200, 255), 	# color初始值
	np.uint8			# 陣列型態
)
m1 = Image.fromarray(m1)
f = ImageFont.truetype("font/msjh.ttc", 20)
#ImageDraw.Draw(PIL圖像變數).text(文字位置,  要寫的文字, 顏色, 設定)
ImageDraw.Draw(m1).text((50,50), "要寫的文字", (255,0,0), f)
m1 =  np.array(m1)



