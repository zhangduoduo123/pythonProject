nums1=[1,2]
n =2
nums2=[3,4]
m = 2
sorted = []
p1, p2 = 0, 0
while p1 < m or p2 < n:
    if p1 == m:
        sorted.append(nums2[p2])
        p2 += 1
    elif p2 == n:
        sorted.append(nums1[p1])
        p1 += 1
    elif nums1[p1] < nums2[p2]:
        sorted.append(nums1[p1])
        p1 += 1
    else:
        sorted.append(nums2[p2])
        p2 += 1
nums1[:] = sorted
print(nums1[:])