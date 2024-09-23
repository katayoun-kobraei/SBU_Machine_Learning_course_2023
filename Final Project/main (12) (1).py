import base64
from io import BytesIO

import numpy as np
from PIL import Image
from rich.theme import Theme
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import ast
import math
import requests
from sklearn.feature_extraction.text import CountVectorizer
import gc
from sklearn.metrics import pairwise_distances
import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
# from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt

# *******************************************************

movies_meta = pd.read_csv("movies_metadata.csv")
check = movies_meta['release_date']
movies_meta = movies_meta[:10000]
copy_movies_meta = movies_meta
credits = pd.read_csv('credits.csv')
copy_credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')
keywords = keywords[:20000]

# *******************************************************

num_missing = movies_meta[movies_meta['production_companies'].isnull()]
clean_movies_meta = movies_meta.dropna(subset=['production_companies'], inplace=True)

movies_meta = movies_meta[movies_meta.id != '1997-08-20']
movies_meta = movies_meta[movies_meta.id != '2012-09-29']
movies_meta = movies_meta[movies_meta.id != '2014-01-01']
movies_meta = movies_meta.astype({'id': 'int64'})

movies_meta = movies_meta.merge(keywords, on='id')
movies_meta = movies_meta.merge(credits, on='id')


def btc_function(data):
    if type(data) == str:
        return ast.literal_eval(data)['name'].replace(" ", "")
    return data


# https://www.kaggle.com/hadasik/movies-analysis-visualization-newbie
def get_values(data_str):
    if isinstance(data_str, float):
        pass
    else:
        values = []
        data_str = ast.literal_eval(data_str)
        if isinstance(data_str, list):
            for k_v in data_str:
                values.append(k_v['name'].replace(" ", ""))
            return str(values)[1:-1]
        else:
            return None


movies_meta['btc_name'] = movies_meta.belongs_to_collection.apply(btc_function)
movies_meta[
    ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'keywords', 'cast', 'crew']] = \
    movies_meta[['genres', 'production_companies', 'production_countries', 'spoken_languages', 'keywords', 'cast',
                 'crew']].applymap(get_values)
movies_meta['is_homepage'] = movies_meta['homepage'].isnull()

movies_meta['status'] = movies_meta['status'].fillna('')
movies_meta['original_language'] = movies_meta['original_language'].fillna('')
movies_meta['btc_name'] = movies_meta['btc_name'].fillna('')

movies_meta.drop_duplicates(inplace=True)


def vector_values(df, columns, min_df_value):
    c_vector = CountVectorizer(min_df=min_df_value)
    df_1 = pd.DataFrame(index=df.index)
    for col in columns:
        print(col)
        df_1 = df_1.join(
            pd.DataFrame(c_vector.fit_transform(df[col]).toarray(), columns=c_vector.get_feature_names_out(),
                         index=df.index).add_prefix(col + '_'))
    return df_1


movies_meta_addon_1 = vector_values(movies_meta,
                                    columns=['status', 'original_language', 'genres', 'production_companies',
                                             'production_countries', 'spoken_languages', 'keywords', 'cast', 'crew'],
                                    min_df_value=20)
movies_meta_addon_2 = vector_values(movies_meta, columns=['btc_name'], min_df_value=2)

col = ['belongs_to_collection', 'genres', 'homepage', 'id', 'imdb_id', 'overview', 'poster_path', 'status',
       'original_language',
       'production_companies', 'production_countries', 'spoken_languages', 'keywords', 'cast', 'crew', 'tagline',
       'adult']
movies_meta.drop(columns=col, inplace=True)
col = ['video', 'is_homepage']
for c in col:
    movies_meta[c] = movies_meta[c].astype(bool)
    movies_meta[c] = movies_meta[c].astype(int)


def get_year(date):
    return str(date).split('-')[0]


movies_meta['popularity'] = movies_meta['popularity'].astype(float)
movies_meta['budget'] = movies_meta['budget'].astype(float)
movies_meta['vote_average_group'] = pd.qcut(movies_meta['vote_average'], q=10, precision=2, duplicates='drop')
movies_meta['popularity_group'] = pd.qcut(movies_meta['popularity'], q=10, precision=2, duplicates='drop')
movies_meta['vote_average_group'] = pd.qcut(movies_meta['vote_average'], q=10, precision=2, duplicates='drop')
movies_meta['runtime_group'] = pd.qcut(movies_meta['runtime'], q=10, precision=2, duplicates='drop')
movies_meta['budget_group'] = pd.qcut(movies_meta['budget'], q=10, precision=2, duplicates='drop')
movies_meta['revenue_group'] = pd.qcut(movies_meta['revenue'], q=10, precision=2, duplicates='drop')
movies_meta['vote_count_group'] = pd.qcut(movies_meta['vote_count'], q=10, precision=2, duplicates='drop')
movies_meta['release_year'] = movies_meta['release_date'].apply(get_year)

movies_meta['release_year'] = movies_meta['release_year'].fillna('')
movies_meta['release_year'] = movies_meta['release_year'].astype(float)
movies_meta['release_year_group'] = pd.qcut(movies_meta['release_year'], q=10, precision=2, duplicates='drop')
movies_meta['title_new'] = movies_meta.apply(lambda x: str(x['title']) + ' (' + str(x['release_date']) + ')', axis=1)

movies_meta_addon_3 = pd.get_dummies(movies_meta[
                                         ['vote_average_group', 'popularity_group', 'runtime_group', 'budget_group',
                                          'revenue_group', 'vote_count_group', 'release_year_group']])
movies_meta_train = pd.concat(
    [movies_meta_addon_1, movies_meta_addon_2, movies_meta_addon_3, movies_meta[['video', 'is_homepage']]], axis=1)

movies_meta_train.index = movies_meta['title_new']
gc.collect()


def get_similar_movies(movie_title, num_rec=10):
    try:
        sample_1 = 1 - pairwise_distances([movies_meta_train.loc[movie_title].values], movies_meta_train.values,
                                          metric='cosine')
        sample_1 = pd.DataFrame(sample_1.T, index=movies_meta_train.index)
        return sample_1.sort_values(by=0, ascending=False).head(num_rec).index
    except ValueError as e:
        print(e)

# *******************  Evaluation Part ******************
def evaluate_recommendations(test_set, get_similar_movies_func, num_rec=10):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for user, items in test_set.items():
        recommended_items = get_similar_movies_func(user, num_rec)
        recommended_items = set(recommended_items)
        relevant_items = set(items)

        true_positives += len(recommended_items.intersection(relevant_items))
        false_positives += len(recommended_items - relevant_items)
        false_negatives += len(relevant_items - recommended_items)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1

test_set = {
    'Toy Story (1995-10-30)': ['Toy Story 2 (1999-10-30)', 'Chicken Run (2000-06-21)'],
    'The Lion King (1994-06-23)': ['Toy Story (1995-10-30)', 'The Little Mermaid (1989-11-17)'],
    'Papillon (1973-12-13)': ['The Godfather (1972-03-14)'],
    'The Godfather (1972-03-14)': ['Papillon (1973-12-13)']
}

# Evaluate the recommendations using the test set
precision, recall, f1 = evaluate_recommendations(test_set, get_similar_movies, num_rec=10)

print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1)

# ************  Using deep Learning Methods *************
# def build_model(input_dim):
#     inputs = Input(shape=(input_dim,))
#     x = Dense(256, activation='relu')(inputs)
#     x = Dropout(0.2)(x)
#     x = Dense(128, activation='relu')(x)
#     x = Dropout(0.2)(x)
#     x = Dense(64, activation='relu')(x)
#     embeddings = Dense(32, activation='relu')(x)
#
#     model = Model(inputs=inputs, outputs=embeddings)
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
#     return model

# Scale the feature data
scaler = StandardScaler()
X = scaler.fit_transform(movies_meta_train.values)

# Create and train the model
# input_dim = X.shape[1]
# model = build_model(input_dim)
# model.fit(X, X, batch_size=64, epochs=30, verbose=1)


# def nn_get_similar_movies(movie_title, num_rec=10):
#     try:
#         movie_embedding = model.predict(scaler.transform([movies_meta_train.loc[movie_title].values]))
#         all_embeddings = model.predict(X)
#         similarities = tf.keras.losses.cosine_similarity(movie_embedding, all_embeddings)
#
#         top_indices = np.argsort(similarities.numpy())[:num_rec]
#         return movies_meta_train.index[top_indices]
#     except ValueError as e:
#         print(e)



# *******************  Front Variables ******************
input1 = ""
input2 = ""
input3 = ""

# movie_name1 = "Undisputed III : Redemption"
# movie_name2 = "Finding Nemo"
# movie_name3 = "Thor"
movie_name1 = ""
movie_name2 = ""
movie_name3 = ""

movie_year1 = 0
movie_year2 = 0
movie_year3 = 0

input1_count = 0
input2_count = 0
input3_count = 0

recomendation_list = []

counter1 = 0
counter2 = 0
counter3 = 0


# *******************  Front Part ******************
# @st.cache_data()
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://raw.githubusercontent.com/katayoun-kobraei/background-image/main/Untitled%20Design%20(6)-fotor-20230706231926.png");
            background-attachment: fixed;
            background-size: 100% 100%;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


add_bg_from_url()


# st.set_page_config(page_title="Movie Recommender System", page_icon="🎥", theme="light")

heading_style = """
    <style>
    .my-heading {
        color: #9B2107;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    </style>
"""
st.markdown(heading_style, unsafe_allow_html=True)
st.markdown("<h1 class='my-heading'>Movie Recommender System</h1>", unsafe_allow_html=True)

select_list = []

for s in movies_meta['title_new'].values:
    s = s.replace('(', "")
    s = s.replace(')', "")
    s = s.split('-')[0]
    inp = s[0:-4]
    inp = inp + " - "
    inp = inp + s[-4:]
    select_list.append(inp)



st.write("""
 **description :** This movie recommendation app is a helpful tool for users who are 
looking for new movies to watch based on their previous viewing preferences. 
It offers a personalized and convenient way for users to discover new movies that
they are likely to enjoy.
***
""")

st.write("""
 ## Enter 3 movies that you have watched and liked with their rates.
""")

movie_name1 = movie_name1 + (st.selectbox('Enter your first movie:',select_list))[0:-8]
movie_rate1 = st.slider(
    f'Rate {movie_name1}!',
    0, 10, 1,
    key='movie_rate1')
st.write("""
***
""")
movie_name2 = movie_name2 + (st.selectbox('Enter your second movie:',select_list))[0:-8]
movie_rate2 = st.slider(
    f'Rate {movie_name2}!',
    0, 10, 1,
    key='movie_rate2')
st.write("""
***
""")
movie_name3 = movie_name3 + (st.selectbox('Enter your third movie:',select_list))[0:-8]
movie_rate3 = st.slider(
    f'Rate {movie_name3}!',
    0, 10, 1,
    key = 'movie_rate3'
)

button_style = """
    <style>
    .my-button {
        background-color: #ff6347;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    </style>
"""
st.markdown(button_style, unsafe_allow_html=True)
if st.button("Suggest me!", key="my-button", help="Submit!"):

    filtered_row3 = movies_meta.loc[
        (movies_meta['original_title'] == movie_name1)]
    input1 = filtered_row3.iloc[0]['title_new']

    filtered_row2 = movies_meta.loc[
        (movies_meta['original_title'] == movie_name2)]
    input2 = filtered_row2.iloc[0]['title_new']

    filtered_row3 = movies_meta.loc[
        (movies_meta['original_title'] == movie_name3)]
    input3 = filtered_row3.iloc[0]['title_new']

    recommend1 = get_similar_movies(input1)[1:]
    recommend2 = get_similar_movies(input2)[1:]
    recommend3 = get_similar_movies(input3)[1:]


    # nn_recommend1 = nn_get_similar_movies(input1)
    # nn_recommend2 = nn_get_similar_movies(input3)
    # nn_recommend3 = nn_get_similar_movies(input3)


    input1_count = math.ceil((movie_rate1 * 10) / (movie_rate1 + movie_rate2 + movie_rate3))
    input2_count = math.ceil((movie_rate2 * 10) / (movie_rate1 + movie_rate2 + movie_rate3))
    input3_count = math.ceil((movie_rate3 * 10) / (movie_rate1 + movie_rate2 + movie_rate3))

    while True:
        if len(recomendation_list) >= 10:
            break
        if counter1 <= input1_count:
            recomendation_list.append([recommend1[counter1], movie_name1])
            counter1 += 1
        if len(recomendation_list) >= 10:
            break
        if counter2 <= input2_count:
            recomendation_list.append([recommend2[counter2], movie_name2])
            counter2 += 1
        if len(recomendation_list) >= 10:
            break
        if counter3 <= input3_count:
            recomendation_list.append([recommend3[counter3], movie_name3])
            counter3 += 1

    for i in range(10):
        movie, based = recomendation_list[i]

        movie_df = movies_meta.loc[
            (movies_meta['title_new'] == movie)]

        original_title = movie_df['original_title'].values[0]
        release_year = int(movie_df['release_year'].values[0])

        movie_metadata_df = copy_movies_meta.loc[
            (copy_movies_meta['original_title'] == original_title)]

        st.write(f"**{i + 1}. {original_title} ({release_year})** based on {based} \n\n")

        p_id = movie_metadata_df['id'].values[0]
        response = requests.get(
            'https://api.themoviedb.org/3/movie/{}?api_key=1c8d419f76f8f9870d3e91eb896c2e54'.format(str(p_id)))
        data = response.json()
        # poster_path = data['poster_path']
        #
        # if data['poster_path'] != "None":
        #     full_path = "https://image.tmdb.org/t/p/w500/" + data['poster_path']
        #     response = requests.get(full_path)
        #     image = Image.open(BytesIO(response.content))
        #     resized_image = image.resize((200, 200))  # Resize the image to 50x50 pixels
        #
        #     # Save the resized image to a new file
        #     resized_image_path = "resized_image.jpg"
        #     resized_image.save(resized_image_path)
        #
        #     # Display the resized image
        #     st.image(resized_image_path)
        if 'poster_path' in data and data['poster_path'] is not None:
            poster_path = data['poster_path']
            full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
            response = requests.get(full_path)
            image = Image.open(BytesIO(response.content))
            resized_image = image.resize((200, 200))  # Resize the image to 200x200 pixels

            # Save the resized image to a new file
            resized_image_path = "resized_image.jpg"
            resized_image.save(resized_image_path)

            # Display the resized image
            st.image(resized_image_path)
        else:
            st.write("No poster image available for this movie.")

        cast_df = copy_credits.loc[
            (copy_credits['id'] == int(p_id))]

        with st.expander(f"See more about {original_title}"):
            runtime = int(movie_metadata_df['runtime'].values[0])
            hour = int(runtime / 60)
            minute = runtime - (hour * 60)
            imdb_rate = movie_metadata_df['vote_average'].values[0]
            popularity = round(movie_metadata_df['popularity'].values[0], 2)
            movie_overview = movie_metadata_df['overview'].values[0]
            movie_tagline = movie_metadata_df['tagline'].values[0]
            vote_count = int(movie_metadata_df['vote_count'].values[0])
            actors_list = cast_df['cast'].values[0]
            actors_list = actors_list.replace("'", '')
            actors_list = actors_list.split('}, {')[0:5]
            actors = []
            characters = []
            for actor in actors_list:
                actor = actor[3:].split(', ')
                for i in actor:
                    if i[:5] == 'name:':
                        actors.append(i[5:])
                        break
                    if i[:10] == 'character:':
                        characters.append(i[10:])

            actors_line = f"**Stars :** {actors[0]} (as {characters[0]})  .  " \
                          f" {actors[1]} (as {characters[1]})  .  " \
                          f" {actors[2]} (as {characters[2]})  .  " \
                          f" {actors[3]} (as {characters[3]})  .  " \
                          f" {actors[4]} (as {characters[4]}) "

            crew_list = cast_df['crew'].values[0]
            crew_list = crew_list.replace("'", '')
            index = crew_list.find("job: Director, name:")
            director = crew_list[index + 20:].split(',')[0]

            genres_list = movie_metadata_df['genres'].values[0]
            genres_list = genres_list[2:-2]
            genres_list = genres_list.replace("'", '')
            genres_list = genres_list.split('}, {')
            movie_genres = ""

            # Extract the genre names from each dictionary
            genre_names = []
            for genre in genres_list:
                genre = genre.split(',')
                if genre[1][7:] == 'Animation':
                    emoji = " 🎨"
                elif genre[1][7:] == 'Comedy':
                    emoji = " 🤡"
                elif genre[1][7:] == 'Family':
                    emoji = " 👨‍👩‍👧‍👦"
                elif genre[1][7:] == 'Crime':
                    emoji = " 🕵️‍"
                elif genre[1][7:] == 'Adventure':
                    emoji = " 🛣️‍"
                elif genre[1][7:] == 'Fantasy':
                    emoji = " 🧚"
                elif genre[1][7:] == 'Romance':
                    emoji = " 💏"
                elif genre[1][7:] == 'Drama':
                    emoji = " 🎭"
                elif genre[1][7:] == "Action":
                    emoji = " 🔫"
                elif genre[1][7:] == 'Thriller':
                    emoji = " 🍿"
                elif genre[1][7:] == 'Horror':
                    emoji = " 😱"
                elif genre[1][7:] == 'History':
                    emoji = " 🏛"
                elif genre[1][7:] == 'Science Fiction':
                    emoji = " 🤖👽"
                elif genre[1][7:] == 'Mystery':
                    emoji = " ❓"
                elif genre[1][7:] == 'War':
                    emoji = " ⚔"
                elif genre[1][7:] == 'Western':
                    emoji = " 🤠"
                elif genre[1][7:] == 'Foreign':
                    emoji = " 🗺"
                elif genre[1][7:] == 'Music':
                    emoji = " 🎶"
                elif genre[1][7:] == 'Documentary':
                    emoji = " 🤔"

                movie_genres = movie_genres + emoji
                movie_genres = movie_genres + genre[1][7:]
                movie_genres = movie_genres + "       "
                genre_names.append(genre[1][7:])

            second_line = f"                                                           " \
                          f"                                          " \
                          f"                                    IMDb RATING          POPULARITY\n"
            third_line = f" {release_year}    {hour}h {minute}m                                      " \
                         f"                                                                   " \
                         f"     ⭐{imdb_rate}/10             " \
                         f"📊 {popularity} ({vote_count})\n \n \n"
            movie_genres = movie_genres.replace(' ', '&nbsp;')
            second_line = second_line.replace(' ', '&nbsp;')
            actors_line = actors_line.replace(' ', '&nbsp;')
            third_line = third_line.replace(' ', '&nbsp;')
            st.header(f"{original_title} \n")
            st.write(second_line, unsafe_allow_html=True)
            st.write(third_line, unsafe_allow_html=True)
            st.write(movie_genres, unsafe_allow_html=True)
            st.write("""
            ***
            """)
            st.markdown(f" ##### {movie_tagline}")
            st.write(movie_overview)
            st.write("""
            ***
            """)
            st.markdown(f"**Director(s) :** {director}")
            st.markdown(actors_line, unsafe_allow_html=True)
