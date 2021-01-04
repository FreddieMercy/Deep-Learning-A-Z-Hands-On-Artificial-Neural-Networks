import pandas as pd

df = pd.read_csv('data/pokemon_data/pokemon_data.csv')
# df_xlsx = pd.read_excel("data/pokemon_data.xlsx")
# print(df_xlsx.tail(10))
df_txt = pd.read_csv('data/pokemon_data/pokemon_data.txt', delimiter='\t')
print(df_txt.head(10))

print(df.tail(10))
df.columns
print(df['Name'])
print(df.Name)
print(df.iloc[1:4])  # stands for "index location". Contrary to default "label location"

print(df.loc[1:4])
print(df.loc[1:4, 'Name'])

df.iloc[2, 1]

for index, row in df.iterrows():
    print(index, row)

print(df.loc[df['Name'] == 'Bulbasaur'])
print(df.describe())

df.sort_values('Name', ascending=False)
df.sort_values(['Name', 'HP'], ascending=False)
df.sort_values(['Name', 'HP'], ascending=[1, 0])  # Aseconding Name, descending HP
df['Total'] = df['HP'] + df['Attack']
print(df)

df['Total2'] = df.iloc[:, 4:10].sum(axis=1)
print(df)

col = list(df.columns)
df = df[col[0:4] + [col[-1]] + col[4:12]]

df.head(5)

df.to_csv('data/pokemon_data/modified.csv')
df.to_csv('data/pokemon_data/modified2.csv', index=False)
df.to_csv('data/pokemon_data/modified3.txt', index=False, sep='\t')
