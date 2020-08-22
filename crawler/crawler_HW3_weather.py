import requests
from bs4 import BeautifulSoup
import prettytable

columns = ["地區", "氣溫"]
prtbl = prettytable.PrettyTable(columns, encoding="utf8")

h = {
	"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36",
	"cookie":"V8_LANG=C; _ga=GA1.3.315247494.1587443680; _gid=GA1.3.1042151494.1594100892; TS01c55bd7=0107dddfefd65d6d626d30df7216be31a86094f75123f2ff49c870749cd19b6d6a5f51803a"
}
url = "https://www.cwb.gov.tw/V8/C/W/TemperatureTop/County_TMax_T.html"

# 天氣-->縣市溫度極值
r1 = requests.get(url,headers=h,
	params={
		"ID":"Tue%20Jul%2007%202020%2017:05:35%20GMT+0800%20(%E5%8F%B0%E5%8C%97%E6%A8%99%E6%BA%96%E6%99%82%E9%96%93)"
	})
bs = BeautifulSoup(r1.text, "html.parser")
rows = bs.find_all("tr")
#print(rows)
prtbl.clear_rows()
for row in rows:
	city = row.find("th").text
	#temp = row.find("span", {"class":"tem-C is-active"}).text
	temp = row.find("span").text

	prtbl.add_row([city, temp])
print(prtbl)
