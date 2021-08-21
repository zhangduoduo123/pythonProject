# import numpy as np
# data = [1,2,3]
# print(data.describe())
# import pandas as pd
# data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
#         'year': [2012, 2012, 2013, 2014, 2014],
#         'reports': [4, 24, 31, 2, 3]}
# df = pd.DataFrame(data, index = ['Cochice', 'Pima', 'Santa Cruz', 'Maricopa', 'Yuma'])
# print(df)
# dft = df.copy(True)
# dff = df.copy(False)
#
# dft = dft.drop(df.index[2])
# dff = dff.drop(df.index[2])
# print(dft)
# print(dff)
a = input()
b = a
b[0] = 100
print(a)