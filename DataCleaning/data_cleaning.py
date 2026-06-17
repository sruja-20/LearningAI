import pandas as pd
import numpy as np

# Read csv file using pandas
df = pd.read_csv('Titanic-Dataset.csv')

# Use inbuilt functions to explore data
df.info()
df.head()

# check for duplicate values
print(df.duplicated())

# identify column data type
cat_col = [col for col in df.columns if df[col].dtype == 'object']
num_col = [col for col in df.columns if df[col].dtype != 'object']

print('Categorical columns:', cat_col)
print('Numerical columns:', num_col)

# Count unique values in the column
u = df[cat_col].nunique()
print(u)

# Missing value as percentage
print(round((df.isnull().sum() / df.shape[0]) * 100, 2))

# handle missing values
df1 = df.drop(columns=['Name', 'Ticket', 'Cabin'])
df1.dropna(subset=['Embarked'], inplace=True)
df1['Age'] = df1['Age'].fillna(df1['Age'].mean())


# Detect outlier with boxplot
import matplotlib.pyplot as plt

plt.boxplot(df1['Age'], vert=False)
plt.ylabel('Variable')
plt.xlabel('Age')
plt.title('Box Plot')
plt.show()

# Calculating outer boundaries & removing them
mean = df1['Age'].mean()
std = df1['Age'].std()

lower_bound = mean - 2 * std
upper_bound = mean + 2 * std

df2 = df1[(df1['Age'] >= lower_bound) & (df1['Age'] <= upper_bound)]


# Finding missing data again
df3 = df2.fillna(df2['Age'].mean())
df3.isnull().sum()

# Recalculate outliers again
mean = df3['Age'].mean()
std = df3['Age'].std()

lower_bound = mean - 2 * std
upper_bound = mean + 2 * std

print('Lower Bound :', lower_bound)
print('Upper Bound :', upper_bound)

df4 = df3[(df3['Age'] >= lower_bound) & (df3['Age'] <= upper_bound)]


# Data validation & verification
X = df3[['Pclass','Sex','Age', 'SibSp','Parch','Fare','Embarked']]
Y = df3['Survived']

# Data formatting
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

num_col_ = [col for col in X.columns if X[col].dtype != 'object']
x1 = X
x1[num_col_] = scaler.fit_transform(x1[num_col_])
x1.head()

print(df3.shape)
print(df.shape)