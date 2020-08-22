
import cv2
import numpy as np

# 運算
# m1 = cv2.imread("image/a2.jpg", 1)
# m2 = np.full(m1.shape, 5, np.uint8)
#m3 = cv2.add(m1, m2)
#m3 = cv2.subtract(m1, m2)
#m3 = cv2.absdiff(m1, m2)
#m3 = cv2.divide(m1, m2)
#m3 = cv2.multiply(m3, m2)
#m3 = cv2.bitwise_not(m1)

# m2 = np.full(m1.shape, 250, np.uint8)
# m4 = np.full(m1.shape, 200, np.uint8)
# m3 = cv2.subtract(m1, m2)
# m3 = cv2.multiply(m3, m4)

# 圖像縮放
# w = 300
# h = int(w/m1.shape[1]*m1.shape[0])
# m2 = cv2.resize(m1, (w,h))
#
# h = 600
# w = int(h/m1.shape[0]*m1.shape[1])
# m3 = cv2.resize(m1, (w,h))
# cv2.imwrite("image/a2.jpg", m3, [cv2.IMWRITE_JPEG_QUALITY, 100])

# 圖像翻轉
# m2 = cv2.flip(m1, -1)
# m3 = cv2.flip(m1, 0)

# 圖像旋轉：
# rot_m = cv2.getRotationMatrix2D((500, 0), 45, 0.5)
# m2=cv2.warpAffine(m1, rot_m, (800,600))

# 區域裁切、複製和貼上
m1 = cv2.imread("image/a2.jpg", 1)
print(m1.shape)
m2 = np.full(m1.shape, 255, np.uint8)
#m2[100:300, 50:100] = m1[400:600, 0:50]
#m2[:, 0:50] = m1[:, 0:50]

#cv2.imshow("Image 1", m1)
cv2.imshow("Image 11", m1[:,:,0]) # B
cv2.imshow("Image 12", m1[:,:,1]) # G
cv2.imshow("Image 13", m1[:,:,2]) # R
#cv2.imshow("Image 2", m1[300:600, 0:200])
#cv2.imshow("Image 2", m2)
#cv2.imshow("Image 3", m3)
cv2.waitKey(0)
cv2.destroyAllWindows()



