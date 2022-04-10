import numpy as np
import pandas as pd
import os
import time
import re
from io import StringIO
import matplotlib.pyplot as plt

fp = os.path.join('./data_mining/movies.dat')
with open(fp, 'r') as file:
    movies = file.read()
fp = os.path.join('./data_mining/ratings.dat')
with open(fp, 'r') as file:
    ratings = file.read()

headers = ['MovieID', 'Title', 'Genres', 'sorted_by_genre']
df_movies = pd.read_csv(StringIO(movies), sep="::", names=headers, engine='python')
headers = ['UserID', 'MovieID', 'Rating', 'Timestamp']
df_ratings = pd.read_csv(StringIO(ratings), sep="::", names=headers, engine='python')

pd.set_option('max_columns', None)
pd.options.display.width = None

genres_list = []
for i in range(len(df_movies)):
    rating_counts = df_movies['Genres'][i].split('|')
    for j in rating_counts:
        if j not in genres_list:
            genres_list.append(j)
genres_list.sort()

df_merged = pd.merge(df_movies, df_ratings)
rating_counts = df_ratings['MovieID'].value_counts()  # counts occurrences of rating for each movie
rating_counts = rating_counts.sort_index()  # sorts from the lowest index, corresponds with df_movies
df_movies = df_movies.join(rating_counts, on='MovieID', lsuffix='_caller')  # appends column with ratings_count
df_movies = df_movies.rename(columns={'MovieID_caller': 'MovieID', 'MovieID': 'ratings_count'})
df_movies = df_movies.sort_values(by='ratings_count', ascending=False)  # sorts by biggest ratings_count
mean_ratings = df_ratings.groupby('MovieID')['Rating'].mean()  # avg rating for each movie
df_movies = df_movies.join(mean_ratings, on='MovieID', lsuffix='_caller')  # appends column with avg ratings
last_timestamps = df_ratings.groupby('MovieID')['Timestamp'].max()      # most recent rating for each movie

for label, value in last_timestamps.items():                        # converts timestamp into desired format
    last_timestamps[label] = time.strftime('%d %b %Y', time.localtime(value))
df_movies = df_movies.join(last_timestamps, on='MovieID', lsuffix='_caller')   # appends column with most recent rating

df_movies[genres_list] = False  # appends column for every genre and assign False everywhere
indexes = df_movies.index
df_copy = df_movies.copy()   # needed copy for assignment (chained-indexing error)
for idx in indexes:     # for every movie assigns True/False according to genres
    yy = df_copy.loc[idx, 'Genres'].split('|')
    for j in yy:
        df_copy.loc[idx, j] = True

df_movies = df_copy      # assigns back to main dataframe

frames = []             # list of dataframes for every genre
mean_values = []
for i in genres_list:
    df = df_movies[df_movies['Genres'].str.contains(i)].copy()  # dataframe for each genre
    mean_values.append(df['Rating'].mean())                      # list of mean values for each genre
    df['sorted_by_genre'] = i
    if len(df) < 100:
        df = df.set_index([pd.Index([x for x in range(len(df))])])      # changing indexes
        pass
    else:
        df = df.iloc[0:100]  # top 100
        df = df.set_index([pd.Index([x for x in range(100)])])
    frames.append(df)

# ********************************************************************* cviceni 03
mean_values = []
for i in genres_list:
    df = df_merged[df_merged['Genres'].str.contains(i)].copy()
    mean_values.append(df['Rating'].mean())
result = pd.DataFrame(mean_values, genres_list, columns=['avg_rating'])     # avg rating for every genre

years = []
title = []
for values in df_movies['Title']:       # makes title without a year
    years.append(int(re.search(r'(\([1-3][0-9]{3})\)', values).group()[1:5]))
    title.append(re.sub(r'(\([1-3][0-9]{3})\)', '', values))

df_decades = pd.DataFrame(title, columns=['Title'])     # new dataframe, separated title and year
df_decades['Year'] = years

intervals = [x for x in range(10, 120, 10)]                 # intervals for bins
intervals_name = [f'{x}.leta' for x in range(10, 100, 10)]
intervals_name.append('00.leta')

df_decades['bins'] = 100-(df_decades['Year'].max()-df_decades['Year'])
df_decades['decades'] = pd.cut(df_decades['bins'], intervals, labels=intervals_name, right=False)  # decades column
df_decades['polovina'] = 'prvni'                                        # column with 'prvni polovina' everywhere

for i in range(len(df_decades)):            # changes needed rows to 'druha polovina'
    if df_decades.iloc[i]['Year'] % 10 >= 5:
        df_decades.iloc[i, df_decades.columns.get_loc('polovina')] = 'druha'

df_decades = df_decades.drop(['bins', 'Year'], axis='columns')      # deletes unwanted columns

# ************************************************************************* cviceni 4

qq = df_ratings['Rating'].value_counts()
plt.bar(qq.index, qq)                       # histogram
plt.xlabel('hodnoceni')
plt.ylabel('pocet hodnoceni')
plt.grid(axis='y')
plt.show()

plt.bar(result.index, result['avg_rating'])
plt.axhline(result['avg_rating'].mean(), xmin=0.05, xmax=0.95, color='r')
plt.text(16, 3.7, f'{round(result["avg_rating"].mean(),3)}')
plt.xticks(rotation=90)
plt.show()

