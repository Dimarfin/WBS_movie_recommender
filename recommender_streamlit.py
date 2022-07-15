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

def find_movie_id(movie_name, df):
    df = df[df['title'].str.contains(movie_name)]
    return df['movieId']

def movie_item_coll_filter(movie_id, ratings, movies, n=10):
    movie_crosstab = pd.pivot_table(data=ratings, 
                                     values='rating', 
                                     index='userId', 
                                     columns='movieId')
    fillna_val=2.51111
    movie_crosstab = movie_crosstab.fillna(value=fillna_val)
    similar_to = movie_crosstab.corrwith(movie_crosstab[movie_id],axis=0)
    similar_to = pd.DataFrame(similar_to, columns=['PearsonR'])
    similar_to.sort_values("PearsonR", ascending=False,inplace=True)
    top_similar = similar_to.head(n+1)
    top_similar = top_similar[top_similar.index != movie_id]
    if len(top_similar.iloc[:,0])>n:
        top_similar = similar_to.head(n)
        
    top_similar = top_similar.merge(movies[['movieId','title','genres']], how='left', on='movieId')
    return top_similar

def movie_user_coll_filter(user_id, ratings, movies, n=10):
    users_items = pd.pivot_table(data=ratings, 
                                 values='rating', 
                                 index='userId', 
                                 columns='movieId')
    
    fillna_val=2.51111
    users_items = users_items.fillna(value=fillna_val)
    similarity = pd.DataFrame(cosine_similarity(users_items),
                           columns=users_items.index, 
                           index=users_items.index)
    
    weights = (similarity.loc[similarity.index!=user_id,user_id]
               /sum(similarity.loc[similarity.index!=user_id,user_id])
               )
    #all users except the one with user_id 
    #and movies not rated by user_id
    not_rated = users_items.loc[users_items.index!=user_id, users_items.loc[user_id,:]==fillna_val]
    weighted_averages = pd.DataFrame(not_rated.T.dot(weights), columns=["predicted_rating"])
    weighted_averages_top = weighted_averages.sort_values(by='predicted_rating', ascending=False).head(n)
    weighted_averages_top = weighted_averages_top.merge(movies[['movieId','title','genres']], how='left', on='movieId')
    return weighted_averages_top

#get genre list
all_listed_genres = list(movies['genres'].str.split('|'))
flat_list = [x for xs in all_listed_genres for x in xs]
genres = sorted(list(set(flat_list)))
genres = ['all']+genres[1:]+[genres[0]]

st.set_page_config(layout="wide") 

# with st.container():
#     video = ('wbsflix_video.mp4')
#     st.video(video)

st.title("🎬 WBSFLIX movie recommender")
st.write("""
By group 5: Dzmitry, Marvin, Weiling, Tamuka 
""")
 
st.write("""
### Popularity based
""")
genre_choise = st.selectbox(
                      'Choose a genre',
                      (genres)
                      )
popul_df = movie_popul_rating(genre_choise, ratings, movies)
#st.text(genre_choise)
st.table(popul_df)

st.write("""
### Similar to a particular movie
""")
movie_name = st.text_input('Movie title or key word', 'Forrest Gump')
ids = find_movie_id(movie_name, movies)

top_similar = pd.DataFrame()
for movie_id in ids:
    top_similar = pd.concat([top_similar, 
                            movie_item_coll_filter(movie_id,ratings, movies ,n=10)
                            ], 
                            axis=0)
top_similar = top_similar.sort_values('PearsonR',ascending=False)
st.table(top_similar.reset_index(drop=True).head(10))

st.write("""
### User based collaborative filtering
""")
user_id = st.number_input('Insert a user ID number', min_value=1, max_value=610, value=100, step=1)
like_user = movie_user_coll_filter(user_id, ratings, movies,n=10)
 
st.table(like_user.head(10))

# Main page 🎈
#🎬