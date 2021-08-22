def fun():
    nums=[2,2]
    target = 4
    hashtable = dict()

    for i, num in enumerate(nums):
        if target - num in hashtable:
            return [hashtable[target - num], i]
        hashtable[nums[i]] = i
    return []
print(fun())