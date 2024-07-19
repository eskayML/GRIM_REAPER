import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
from typing import List 



model = SentenceTransformer('all-MiniLM-L6-v2')
df_new = pd.read_csv('small_df.csv')

@st.cache_resource
def load_embeddings(file_path):
    return np.load(file_path)


embeddings = load_embeddings('small_embeds.npy')

def get_embeddings(query:str) -> np.array:
    return model.encode(query)
    

def get_similarity(target: List ,  candidates: List[List[float]]):

  candidates = np.array(candidates)
  target = np.expand_dims(np.array(target),axis=0)


  sim = cosine_similarity(target,candidates)
  sim = np.squeeze(sim).tolist()
  sort_index = np.argsort(sim)[::-1]
  sort_score = [sim[i] for i in sort_index]
  similarity_scores = zip(sort_index,sort_score)

  # Return similarity scores
  return similarity_scores

def search(new_query:str) -> pd.DataFrame:
  # Get embeddings of the new query
  new_query_embeds = get_embeddings([new_query])[0]
  top_recommendations = list(get_similarity(new_query_embeds, embeddings))[:10]
  print(top_recommendations)
  returned_listings = [ df_new.iloc[i[0]] for i in top_recommendations ]
  return pd.DataFrame(returned_listings)




st.title("GRIM REAPER: Predict when you die!💀")
st.image('grim reaper.jpg', width = 150 )

# Input fields
name = st.text_input('Enter your name')
gender = st.radio('Enter Gender', options=['Male', 'Female', 'Other'])
country = st.text_input('Enter your Country')
job_description = st.text_area('Briefly describe your day to day Occupation (the more descriptive ,  the better)', height=100)
current_age = st.number_input("Enter your current age", min_value=0, max_value=120, step=1)
exercise_frequency = st.slider('How frequently do you Exercise? (1 - Very Rarely, 5 - Very Frequently)', min_value=1, max_value=5, value=3)



if st.button("Predict 💀"):
    EXERCISE_WEIGHTS_MAPPING = [0.75, 0.9, 1, 1.2, 1.5]
    this_year = datetime.datetime.now().year
    ordered_inputs = [name, job_description, gender, country, this_year - current_age ]
    mashed = '; '.join(list(map(str, ordered_inputs)))
    output = search(mashed)
    death_age = output['Age of death'].astype(int).median()
    print(death_age)

    threshold = EXERCISE_WEIGHTS_MAPPING[exercise_frequency-1] if death_age > current_age else 1.4
    death_age *= threshold

    death_cause = output['Manner of death'].iloc[:10].value_counts().index[0]

    st.write('You will likely die at age', round(death_age))
    st.write('Likely Cause', death_cause)


st.warning("The predictions here are highly probabilistic and are mostly due to the patterns found in a dataset, It's mostly for educational purposes and  may or may not be when you actually die")
