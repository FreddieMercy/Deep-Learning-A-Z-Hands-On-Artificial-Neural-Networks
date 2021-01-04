def pandasPractice():
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

    arr = [1, 2, 3, 4, 5]
    import numpy as np
    arr2 = np.array([6, 7, 8, 9, 10])
    s1 = pd.Series(arr)
    s2 = pd.Series(arr2)
    s1 = s1.append(s2)
    s1.index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

    s1 = s1.drop('j')

    print(s1)

    arr.append(11)
    print(arr)
    s1 = pd.Series(arr)
    s2 = pd.Series(arr2)

    s1.add(s2)
    s1.sub(s2)
    s1.mul(s2)
    s1.div(s2)

    print("median", s1.median())
    print("max", s1.max())
    print("min", s1.min())

    dates = pd.date_range('today', periods=6)
    print(dates)

    rnd_num_arr = np.random.randn(6, 4)  # randn: in array 6 by 4
    print(rnd_num_arr)

    column = ['A', 'B', 'C', 'D']

    df1 = pd.DataFrame(rnd_num_arr, index=dates, columns=column)

    df1

    data = {
        'animal':
            ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],

        'age':
            [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],

        'visits':
            [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'priority':
            ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']
    }

    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

    df2 = pd.DataFrame(data, index=labels)

    df2

    df2.T  # T stands for traversal?

    df3 = df.copy()  # deep copy, otherwise it will pass by reference
    df3.isnull()  # check where is null
    # df3.fillna('Freddie') # fill wherever is null to 'Freddie'
    df3.dropna(how='any')  # drop any row has missing data

    df2.cumsum()  # return cumulative sum
