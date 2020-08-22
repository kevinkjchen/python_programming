import requests

resp = requests.get("http://teaching.bo-yuan.net/test/requests/")
#print(resp.encoding)
resp.encoding = "utf8"
print(resp.text)

resp = requests.get("http://teaching.bo-yuan.net/test/requests/",
	params={"action": "test1"}
	)
#print(resp.encoding)
resp.encoding = "utf8"
print(resp.text)

resp = requests.delete("http://teaching.bo-yuan.net/test/requests/",
	params={"action": "test1"},
	data={"id": "test2"}
	)
resp.encoding = "utf8"
print(resp.text)

resp = requests.put("http://teaching.bo-yuan.net/test/requests/",
	params={"action": "test1"},
	data={	"id": "test2",
			"name": "test3"
		}
	)
resp.encoding = "utf8"
print(resp.text)

resp = requests.patch("http://teaching.bo-yuan.net/test/requests/",
	params={"action": "test1"},
	data={	"id": "test2",
			"name": "test3",
			"address":"test4"
		}
	)
resp.encoding = "utf8"
print(resp.text)

resp = requests.post("http://teaching.bo-yuan.net/test/requests/",
	params={"action": "test1"},
	data={	"id": "test2",
			"name": "test3",
			"address":"test4"
		}
	)
resp.encoding = "utf8"
print(resp.text)
