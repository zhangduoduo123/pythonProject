def maxSubArray(nums):
    if len(nums) == 1:
        return nums[0]
    dp = [0] * len(nums)

    temp = -1110
    for i in range(len(nums)):

        if i == 0:
            dp[i] = nums[i]
        else:
            dp[i] = max(dp[i - 1] + nums[i], nums[i])
            # temp = max(temp,dp[i])
    return max(dp)
nums=[-2,1,-3,4,-1,2,1,-5,4]
maxSubArray(nums)