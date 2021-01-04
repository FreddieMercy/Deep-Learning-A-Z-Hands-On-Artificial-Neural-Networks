import pandas as pd


def pandasPractice():
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

    new_df = df.loc[(df['Type 1'] == 'Grass') & (df['Type 2'] == 'Poison') | (df['HP'] > 70)].sort_values('HP',
                                                                                                          ascending=False).reset_index(
        drop=True)

    print(new_df)

    new_df = df.loc[(df['Type 1'] == 'Grass') & (df['Type 2'] == 'Poison') | (df['HP'] > 70)].sort_values('HP',
                                                                                                          ascending=False).reset_index(
        drop=True, inplace=True)

    print(new_df)
    print(df)

    df.loc[df['Name'].str.contains("Mega")]
    df.loc[df['Name'].str.contains("Me.a")]

    import re

    df.loc[df['Name'].str.contains("Me.a", flags=re.I, regex=True)]  # flags=re.I means ignore case

    df.loc[df['Type 1'] == 'Fire', 'Legendary'] = True  # change 'Legendary' of all where 'Type 1' is equal to 'Fire'
    df.loc[df['Type 2'] == 'Grass', ['Type 2', 'Legendary']] = ['Fire', True]

    print(df)

    df.groupby(['Type 1']).mean().sort_values('Defense', ascending=False)

    df['count'] = 1
    df.groupby(['Type 1', 'Legendary']).count()['count']

    df['sum'] = 1
    df.groupby(['Type 1', 'Legendary']).sum()

    for df in pd.read_csv('data/pokemon_data/pokemon_data.csv', chunksize=5):  # 5 rows per time
        print(df)
        # break
