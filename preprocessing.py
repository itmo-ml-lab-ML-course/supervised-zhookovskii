import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/books.csv')

scaler = StandardScaler()

columns_to_normalize = df.columns[
    (df.columns != 'Rating') &
    (df.columns != 'Title') &
    (df.columns != 'Author') &
    (df.columns != 'Publisher') &
    (df.columns != 'Language')
]

df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
df[columns_to_normalize] = df[columns_to_normalize].fillna(df[columns_to_normalize].mean())

df.to_csv('data/scaled_books.csv', index=False)