#import requests

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


# def generate_fibonacci(n):
#     a, b = 0, 1
#     result = []
#     while a < n:
#         result.append(a)
#         a = b
#         b = a + b
#         # a, b = b, a + b
#     return result

# print(generate_fibonacci(50))




# def find_occurrences(arr, target):
#     """Find all occurrences of a target in a given array."""
#     occurrences = []
#     for i, num in enumerate(arr):
#         if num == target:
#             occurrences.append(i)
#     return occurrences if occurrences else []


# arr = [1, 2, 3, 4, 5, 6, 7, 5, 9, 10]
# x = 5
# print(find_occurrences(arr, x))




# def ls(arr, x):
#   occurrences = []
#   for i in range(len(arr)):
#     if arr[i] == x:
#       occurrences.append(i)
#       #print("occurrences", occurrences)
#       first_target_index= occurrences[0]
#       #print("first_target_index", first_target_index) #first_target_index
#   if occurrences:
#     return first_target_index, occurrences, len(occurrences)
#   return -1, occurrences






# arr = [1, 2, 3, 4, 5, 6, 7, 5, 9, 1]
# x = 1
# print(ls(arr, x))

fib=[0,1,1,2,3,5,8,13]
idx=[0,1,2,3,4,5,6,7]


def fib(idx):
    if idx <= 1:
        return idx
    return fib(idx-1)+fib(idx-2)
    


print(fib(0))



def ls (arr,x):
    i=0
    if len(arr) == 0:
        return -1
    if arr[0] == x:
        return True, arr[0], i
    else:
        i+=1
    return ls(arr[1:], x)


def bs (arr,x):
    l, r = 0, len(arr)-1
    while l <= r:
        h = (l+r)//2
        if x == arr[h]:
            return True, h
        elif x > arr[h]:
            l = h + 1
        else:
            r = h - 1
    return -1


def rbs(arr, x):
    l=len(arr)
    if l == 0:
        return -1
    h =l//2
    if arr[h]==x:
        return  x, True
    elif arr[h] > x:
        return rbs(arr[:h], x)
    else:
        return rbs(arr[(h+1):], x)




arr=[1,2,3,4,5]
# print(arr[2:])
x=6
print(rbs(arr, x))
