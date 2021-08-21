x = -120
if x < 0:
    tag = -1
    x = -x
res = []
temp = -1
while x != 0:
    temp = x%10  # 3
    res.append(temp)
    x = (x-temp)//10
temp = 0
cnt = 0
for i in res[::-1]:
    temp = temp + i*10**cnt
    cnt = cnt +1
if tag == -1:
    print(-temp)
else:
    print(temp)




