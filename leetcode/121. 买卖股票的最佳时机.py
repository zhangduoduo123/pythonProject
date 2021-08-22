def maxProfit(prices):
    n = len(prices)
    if n == 0: return 0  # 边界条件
    dp = [0] * n
    minprice = prices[0]

    for i in range(1, n):
        minprice = min(minprice, prices[i])
        dp[i] = max(dp[i - 1], prices[i] - minprice)

    return dp[-1]