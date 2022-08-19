import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier


st.write("""
# Mythical creatures prediction App

This App predicts mythical creatures!!!

""")

st.sidebar.header('User Input Features')


st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

#collect user input features into a dataframe

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        home = st.sidebar.selectbox('home', ('Sesame Street', 'Ireland', 'World War Z'))
        gender = st.sidebar.selectbox('gender', ('male', 'female'))
        nose_length_mm = st.sidebar.slider('nose lenght (mm)', 32.1, 59.6, 43.9)
        nose_depth_mm = st.sidebar.slider('nose depth (mm)', 13.1, 21.5, 17.2)
        nose_width_mm = st.sidebar.slider('nose width (mm)', 172.0, 231.0, 210.0)
        weight_kg = st.sidebar.slider('Weight (kg)', 27, 63, 42)
        data = {'home': home,
               'nose_length_mm': nose_length_mm,
               'nose_depth_mm': nose_depth_mm,
               'nose_width_mm': nose_width_mm,
               'weight_kg': weight_kg,
               'gender': gender}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()


#combine user input features with entire dataset

creatures_raw = pd.read_csv('Mythical_creatures.csv')

creatures = creatures_raw.drop(columns=['type'])


df = pd.concat([input_df, creatures], axis=0)

# perform one-hot-encoding on 'gender' and 'home' columns

encode = ['gender', 'home']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

# select only the first row (the user input data)

df = df.iloc[:1]


# Displays the user input features

st.subheader('User Input Features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)


# Reads in saved classification model

load_clf = pickle.load(open('mythical_creatures.pkl', 'rb'))


# Apply model to make predictions

prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
creatures_type = np.array(['Vampire', 'Banshee', 'Zombie'])
st.write(creatures_type[prediction])


st.subheader('Prediction Probability')
st.write(prediction_proba)

