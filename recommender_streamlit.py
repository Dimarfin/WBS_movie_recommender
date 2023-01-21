import streamlit as st
import pandas as pd
import sklearn as skl
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import requests

path = ''
API_key = 'd06b517b45b8643d8cdb739e04465106'

ratings = pd.read_csv(path+'ratings.csv')
movies = pd.read_csv(path+'movies.csv')
links = pd.read_csv(path+'links.csv')

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

st.title("🎬 WBSFLIX movie recommender")
st.markdown("<b style='text-align: right; color: grey;'>Created by Dzmitry Afanasenkau</b>", unsafe_allow_html=True)
 
st.write("""
### Popularity based
""")
genre_choise = st.selectbox(
                      'Choose a genre',
                      (genres),
                      index=3
                      )
popul_df = movie_popul_rating(genre_choise, ratings, movies)
popul_df = popul_df.merge(links[['movieId','imdbId']], how='left', on='movieId')

temp=[]
width = [3]*11
temp[0:10] = st.columns(width)
for i, row in popul_df.head(10).iterrows():
    with temp[i]:
        real_imdbId = '00000000' + str(row['imdbId'])
        real_imdbId = 'tt' + real_imdbId[-7:]
        response = requests.get(f"""https://api.themoviedb.org/3/find/{real_imdbId}?api_key={API_key}&language=en-US&external_source=imdb_id""")
        try:
            st.image(('https://image.tmdb.org/t/p/w92' + response.json()['movie_results'][0]['poster_path']), caption=row['title'], width=None, use_column_width='always', clamp=False, channels="RGB", output_format="auto")
        except:
            st.image(('rsz_movie_default.jpg'), caption=row['title'], width=None, use_column_width='always', clamp=False, channels="RGB", output_format="auto")

df = popul_df
df['votes'] = popul_df['count']
with st.expander("More info"):
    st.table(df[['title','rating','votes','score','genres','imdbId']])        

st.write("""
### Similar to a particular movie
""")
movie_name = st.text_input('Movie title or a part of it', 'Star Wars')
ids = find_movie_id(movie_name, movies)
if not ids.empty:
    movies_with_ids = movies.loc[movies['movieId'].isin(ids)]

    title_1m = st.selectbox(
        'Refine your search',
        tuple(movies_with_ids['title'])
        )
    id_1m = movies_with_ids.loc[movies_with_ids['title'] == title_1m]['movieId']
    
    top_similar_1m = movie_item_coll_filter(int(id_1m), ratings, movies ,n=10)
    top_similar_1m = top_similar_1m.merge(links[['movieId','imdbId']], how='left', on='movieId')
    
    temp=[]
    width = [3]*11
    temp[0:10] = st.columns(width)
    for i, row in top_similar_1m.head(10).iterrows():
        with temp[i]:
            real_imdbId = '00000000' + str(row['imdbId'])
            real_imdbId = 'tt' + real_imdbId[-7:]
            response = requests.get(f"""https://api.themoviedb.org/3/find/{real_imdbId}?api_key={API_key}&language=en-US&external_source=imdb_id""")
            try:
                st.image(('https://image.tmdb.org/t/p/w92' + response.json()['movie_results'][0]['poster_path']), caption=row['title'], width=None, use_column_width='always', clamp=False, channels="RGB", output_format="auto")
            except:
                st.image(('rsz_movie_default.jpg'), caption=row['title'], width=None, use_column_width='always', clamp=False, channels="RGB", output_format="auto")
    
    with st.expander("More info"):
        st.table(top_similar_1m.reset_index(drop=True).head(10)[['title','PearsonR','genres']])
else:
    st.markdown("<b style='text-align: right; color: red;'>No movies found for the given keyword(s)</b>", unsafe_allow_html=True)
    st.write('_')
    st.write('_')
    st.write('_')
    
    
st.write("""
### User based collaborative filtering
""")
user_id = st.number_input('Insert a user ID number', min_value=1, max_value=610, value=100, step=1)
like_user = movie_user_coll_filter(user_id, ratings, movies,n=10)
like_user = like_user.merge(links[['movieId','imdbId']], how='left', on='movieId')

temp=[]
width = [3]*11
temp[0:10] = st.columns(width)
for i, row in like_user.head(10).iterrows():
    with temp[i]:
        real_imdbId = '00000000' + str(row['imdbId'])
        real_imdbId = 'tt' + real_imdbId[-7:]
        response = requests.get(f"""https://api.themoviedb.org/3/find/{real_imdbId}?api_key={API_key}&language=en-US&external_source=imdb_id""")
        try:
            st.image(('https://image.tmdb.org/t/p/w92' + response.json()['movie_results'][0]['poster_path']), caption=row['title'], width=None, use_column_width='always', clamp=False, channels="RGB", output_format="auto")
        except:
            st.image(('rsz_movie_default.jpg'), caption=row['title'], width=None, use_column_width='always', clamp=False, channels="RGB", output_format="auto")

with st.expander("More info"): 
    st.table(like_user.head(10))

# Main page 🎈
#🎬