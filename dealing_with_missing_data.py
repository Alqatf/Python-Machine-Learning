# Python Machine Learning
# Dealing with Missing Data

import pandas as pd
from io import StringIO

# create sample data with missing values
csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
0.0,11.0,12.0,'''

csv_data = unicode(csv_data)
df = pd.read_csv(StringIO(csv_data))

# count the number of missing values in each column
df.isnull().sum()

df.values # access numpy array that underlies pandas df (need np arrays for sklearn)
type(df)
type(df.values)


