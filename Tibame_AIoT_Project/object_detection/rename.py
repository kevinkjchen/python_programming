# 自動把蒐集的資料的檔名作編號
import os
path='FamilyMart/capture/' 
#path = input("輸入路徑:")
base = input("輸入base檔名 [1]1000, [2]2000,...:")

files=os.listdir(path)
print('files') #印出讀取到的檔名稱，用來確認自己是不是真的有讀到

n=0 #設定初始值
for i in files: 
	oldname=path+files[n] #指出檔案現在的路徑名稱，[n]表示第n個檔案
	#newname=path+'img_1{:03d}.jpg'.format(n+1) 
	newname=path+'img_'+'{}{:03d}.jpg'.format(base, n) 
	os.rename(oldname,newname)
	print(oldname+'>>>'+newname) #印出原名與更名後的新名，可以進一步的確認每個檔案的新舊對應
	n=n+1 #當有不止一個檔案的時候，依次對每一個檔案進行上面的流程，直到更換完畢就會結束