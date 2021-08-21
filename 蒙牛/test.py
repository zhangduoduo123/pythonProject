# import numpy as np
# import pandas as pd
# X = [[238.14, 238.14], [190.34, 190.34]]
# # d1 = 1-np.dot(X[0], X[1]) / (np.linalg.norm(X[0]) * np.linalg.norm(X[0]))
# # print(d1)
# x0 = np.mean(X[0])
# x1 = np.mean(X[1])
# X[0]=X[0]-x1
# X[1]=X[1]-x1
# corr = 1-np.dot(X[0], X[1]) / (np.linalg.norm(X[0]) * np.linalg.norm(X[0]))
# print(corr)
# prices = [7,1,5,3,6,4]
# prices=prices[::-1]
# l = len(prices)
# max = 0
# for i in range(l):
#     for j in range(i+1,l):
#         if prices[i]-prices[j] > max:
#             max = prices[i]-prices[j]
# if max < 0:
#     max = 0
# print(max)
# from math import sqrt
#
# from pip._vendor.distlib.compat import raw_input
#
# nums = raw_input()
# nums = nums.split()
# sum = 0
# for i in range(int(nums[1])):
#     sum = sum + float(nums[0])
#     t = sqrt(float(nums[0]))
#     nums[0] = t
#
# print(sum)
# nums=[[1,2,3],[3,4,6]]
# m, n = len(nums), len(nums[0])
# c = 2
# r = 3
#
# ans = [[0] * c for _ in range(r)]
# for x in range(m * n):
#     ans[x // c][x % c] = nums[x // n][x % n]
#
# print(ans)
# board = [["5","3",".",".","7",".",".",".","."]
# ,["6",".",".","1","9","5",".",".","."]
# ,[".","9","8",".",".",".",".","6","."]
# ,["8",".",".",".","6",".",".",".","3"]
# ,["4",".",".","8",".","3",".",".","1"]
# ,["7",".",".",".","2",".",".",".","6"]
# ,[".","6",".",".",".",".","2","8","."]
# ,[".",".",".","4","1","9",".",".","5"]
# ,[".",".",".",".","8",".",".","7","9"]]
#
#
# row = [[0] * 10 for _ in range(9)]
# col = [[0] * 10 for _ in range(9)]
# box = [[0] * 10 for _ in range(9)]
# for i in range(9):
#     for j in range(9):
#         if board[i][j] == '.':
#             continue
#         curNum = ord(board[i][j]) - ord('0')
#         if row[i][curNum] != 0 or col[j][curNum] != 0 or box[j // 3 + (i // 3) * 3][curNum] != 0:
#             print(False)
#         row[i][curNum], col[j][curNum], box[j // 3 + (i // 3) * 3][curNum] = 1, 1, 1
# print(True)

a = 10 # a是整数
print('11/3 = ',11/3)
print('11/3 = ',11/3)
print('11//3 = ',11//3)
print('10%3 = ',10%3)