import streamlit as st
import pandas as pd
import sklearn as skl
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

path = ''

ratings=pd.read_csv(path+'ratings.csv')
movies=pd.read_csv(path+'movies.csv')

def movie_popul_rating(genre, ratings, movies, n=10):
    # genre = 'Comedy'
    # n=10
        
    df = ratings.merge(movies[['movieId','genres']], how='left', on='movieId')
    if genre!='all':
        df = df.loc[df['genres'].str.contains(genre),:]
    
    df = df[['movieId', 'rating']]
    df1 = df.groupby('movieId').mean()
    df1['count'] = df.groupby('movieId').count()
    
    scaler = MinMaxScaler((0,1))
    df1_scaled = scaler.fit_transform(df1)
    df1_scaled = pd.DataFrame(df1_scaled, columns=df1.columns, index=df1.index)

    df1['score'] = df1_scaled['rating']*df1_scaled['count']
    
    df2 = df1.sort_values("score", ascending=False).head(n)
    df3 = df2.merge(movies[['movieId','title','genres']], how='left', on='movieId')
 
    return df3

#get genre list
all_listed_genres = list(movies['genres'].str.split('|'))
flat_list = [x for xs in all_listed_genres for x in xs]
genres = sorted(list(set(flat_list)))
genres = ['all']+genres[1:]+[genres[0]]
 
st.title("Movie recommender v0.04")
 
st.write("""
### Popularity based
""")
genre_choise = st.selectbox(
                      'Which genre would you like?',
                      (genres)
                      )
popul_df = movie_popul_rating(genre_choise, ratings, movies)
#st.text(genre_choise)
st.table(popul_df)



