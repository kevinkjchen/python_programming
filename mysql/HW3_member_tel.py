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
	print("\t", "(5)", "新增會員的電話")
	print("\t", "(6)", "刪除會員的電話")

key = ''
os.system("cls")
m_columns = ["編號", "姓名", "生日", "地址", "電話"]
m_tbl = prettytable.PrettyTable(m_columns, encoding="utf8")
t_columns = ["編號", "電話"]
t_tbl = prettytable.PrettyTable(t_columns, encoding="utf8")

# 連結資料庫
conn = pymysql.connect(
	host="localhost",
	user="root",
	passwd=input("輸入資料庫root密碼:"),
	db="python_ai",
	charset="utf8",
	port=int(input("輸入資料庫的port:"))
)
curs = conn.cursor()

# 顯示資料表
def show_table():
	global curs, m_tbl, t_tbl

	os.system("cls")	
	curs.execute("SELECT `a`.*,`b`.`tel` FROM `member` AS `a` LEFT JOIN `tel` AS `b` ON `a`.`id`=`b`.`member_id` ")
	r = curs.fetchall()
	m_tbl.clear_rows()
	last_id = -1
	for d in r:
		if last_id != d[0]:
			m_tbl.add_row(d)
		else:
			# 如果和上一筆內容一樣則不重複填入資料
			m_tbl.add_row(["", "", "", "", d[4]])		
		last_id = d[0]
	print(m_tbl)	

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
	if key == '5':	
		show_table()
		curs.execute("INSERT INTO `tel`(`member_id`, `tel`) VALUES(%(i)s, %(n)s) ",
			{
				"i":input("請選擇要添加電話的會員編號:"),
				"n":input("請輸入電話:")
			})
		conn.commit()
	if key == '6':	
		show_table()
		
		# 查詢並印出所選會員的所有電話號碼
		curs.execute("SELECT `id`, `tel` FROM `tel` WHERE `member_id`=%s",
			input("請選擇要刪除電話的會員編號:")
			)
		r = curs.fetchall()
		t_tbl.clear_rows()
		for d in r:
			t_tbl.add_row([d[0], d[1]])
		print(t_tbl)

		curs.execute("DELETE FROM `tel` WHERE `id`=%s", 
			input("請選擇要刪除的電話編號:")
			)
		conn.commit()		

	show_manual()
	key = input("指令:")
			