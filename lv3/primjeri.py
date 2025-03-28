import pandas as pd
import numpy as np

"""
s1 = pd.Series(['crvenkapica', 'baka', 'majka', 'lovac', 'vuk'])
print(s1)
s2 = pd.Series(5., index=['a','b','c','d','e'], name = 'ime_objekta')
print(s2)
print(s2['b'])
s3 = pd.Series(np.random.randn(5))
print(s3)
print(s3[3])"

"""

data = pd.read_csv('data_C02_emission.csv')
 #konvertiranje kategorickih velicina u tip category
print(len(data))
print(data)
print(data.head(5))
print(data.tail(3))
print(data.info())
print(data.describe())
print(data.max())
print(data.min())