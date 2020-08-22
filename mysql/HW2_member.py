import os
import codecs
import prettytable
import pymysql

def show_manual():
	print("")
	print("工作路徑:", os.getcwd())
	print("\t", "(0)", "離開程式")
	print("\t", "(1)", "顯示會員資料")
	print("\t", "(2)", "新增會員資料")
	print("\t", "(3)", "更新會員資料")
	print("\t", "(4)", "刪除會員資料")

key = ''
os.system("cls")
columns = ["編號", "姓名", "生日", "地址"]
prtbl = prettytable.PrettyTable(columns, encoding="utf8")

conn = pymysql.connect(
	host="localhost",
	user="root",
	passwd=input("輸入資料庫root密碼:"),
	db="python_ai",
	charset="utf8",
	port=int(input("輸入資料庫的port:"))
)
curs = conn.cursor()

def show_table():
	global curs, prtbl

	os.system("cls")	
	curs.execute("SELECT * FROM `member`")
	r = curs.fetchall()
	prtbl.clear_rows()
	for d in r:
		prtbl.add_row(d)	
	print(prtbl)	

while key != '0':
	
	if key == '1':	
		show_table()
		#print(curs.lastrowid, type(curs.lastrowid))

	if key == '2':
		os.system("cls")	
		curs.execute("INSERT INTO `member`(`name`, `birthday`, `address`)" + 
			" VALUES(%s, %s, %s)",
			[
			input("姓名:"),
			input("生日:"),
			input("地址:")]									
			)	
		conn.commit()
	if key == '3':	
		show_table()
		curs.execute("UPDATE `member` SET `name`=%(n)s, `birthday`=%(d)s, `address`=%(a)s"
			" WHERE `id`=%(i)s",
			{
				"i":input("輸入要修改的編號:"),
				"n":input("姓名:"),
				"d":input("生日:"),
				"a":input("地址:")
			})
		conn.commit()	
	if key == '4':	
		show_table()
		curs.execute("DELETE FROM `member` WHERE `id`=%s", input("輸入要刪除的編號:"))
		conn.commit()

	show_manual()
	key = input("指令:")
			