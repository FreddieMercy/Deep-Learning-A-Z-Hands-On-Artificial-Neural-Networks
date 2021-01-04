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
