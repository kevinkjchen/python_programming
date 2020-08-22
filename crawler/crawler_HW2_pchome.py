import requests
import json
import prettytable

columns = ["名稱", "價格"]
prtbl = prettytable.PrettyTable(columns, encoding="utf8")
keyword = input("關鍵字:")
page = "1"

while page != "0":
	try:
		r1 = requests.get(
			"https://ecshweb.pchome.com.tw/search/v3.3/all/results?q=" + keyword + "&page=" + page + "&sort=sale/dc",
			headers={
				"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36"
				})
	except:
		print("page not found!")	
		break
	#print(r1.text)
	try:
		jn = json.loads(r1.text)
		#jn = json.load(r1)
		#jn = resp.json()
	except:
		print("json decode error!")		
		break

	prods = jn["prods"]
	prtbl.clear_rows()
	for prod in prods:
		prtbl.add_row([prod["name"], prod["price"]])
	print(prtbl)
	page = input("前往頁碼:")
