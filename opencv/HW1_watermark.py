import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

name = input("請輸入圖片檔名：")
text = input("請輸入浮水印內容：")
size = int(input("請輸入浮水印尺寸(px)："))

img = cv2.imread("image/" + name, 1)
print(img.shape)
# height = img.shape[0]
# weight = img.shape[1]

img_pil = Image.fromarray(img)
f = ImageFont.truetype("font/msjh.ttc", size)
ImageDraw.Draw(img_pil).text((10,10), text, (255,0,0), f)
img =  np.array(img_pil)
cv2.imshow("Image 1", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
