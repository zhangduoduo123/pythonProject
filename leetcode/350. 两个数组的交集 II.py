
def intersect(nums1, nums2):
    s = []
    # nums1 = nums1.sort()
    # nums2 = nums2.sort()
    nums1=sorted(nums1)
    nums2 = sorted(nums2)
    m = len(nums1)
    n = len(nums2)
    p1, p2 = 0, 0
    while p1 < m and p2 < n:
        if nums1[p1] == nums2[p2]:
            s.append(nums2[p2])
            p2 += 1
            p1 += 1
        elif nums1[p1] < nums2[p2]:
            p1 += 1
        else:
            p2 += 1
    return s

nums1 = [4,9,5]
nums2 = [9,4,9,8,4]


print(intersect(nums1, nums2))