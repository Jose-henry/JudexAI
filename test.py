import requests

# url = "https://legalpediaresources.com/api/login"

# payload={'email': "imoh.silas@gmail.com",
# 'password': "Lextech2024"}
# files=[

# ]
# headers = {}

# response = requests.request("POST", url, headers=headers, data=payload, files=files)

# print(response.text)




""" url1 = "https://legalpediaresources.com/api/v1/judgements"

payload1 = "{\r\n    \"api_key\": \"P4IDKDvrMUgwYu8rY3PhxxvKG2xxQ9AQ\",\r\n    \"search_param\": \"\", // search by title, search by suit_no, search by judgement_date, search by court \r\n    \"filterByYear\": \"\", // 2024\r\n    \"filterBySbjMatterIndex\": \"\", // ACTION, BANKING\r\n    \"filterByNoSummary\": \"\" // no_summarry\r\n}"
headers = {
  'Accept': 'application/json',
  'Authorization': 'Bearer 107|izBQnuGtkQhj45cejSQVUHOG4xBryfkx4LOG7lBj'
}

response = requests.request("POST", url1, headers=headers, data=payload1)

print(response.text)
 """


def generate_fibonacci(n):
    a, b = 0, 1
    result = []
    while a < n:
        result.append(a)
        a = b
        b = a + b
        # a, b = b, a + b
    return result

print(generate_fibonacci(50))