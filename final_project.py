import streamlit as st                  # pip install streamlit
import pandas as pd
from model.podcast_predictor import PodcastPredictor, get_suggestion
import os

st.set_page_config(page_title="Podcast Listening Time Predictor", layout="wide")

st.markdown("# Podcast Listening Time Predictor")

st.markdown("""
This application helps podcast creators predict the average listening time for their episodes 
based on various metadata factors. Simply input your episode details below to get a prediction.
""")

if 'predictor' not in st.session_state:
    st.session_state.predictor = PodcastPredictor()

    model_path = 'model/podcast_model.joblib'  # TODO: load model
    if os.path.exists(model_path):
        st.session_state.predictor.load_model(model_path)

# Create the input form
st.markdown("## 🎙️ Podcast Info")

with st.form("podcast_form"): # TODO:Frontend Code
    # Episode name (for user reference only)
    episode_name = st.text_input("Episode Name")
    
    # Time Length
    time_length = st.text_input("Time Length (HH:MM or MM:SS)", placeholder="02:00")
    
    # Genre selection
    genres = ["Comedy", "News", "Education", "Business", "Technology", "Health", "Arts", "Sports", "Music", "Society"]
    genre = st.selectbox("Genre", genres)
    
    # Publication day
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    publication_day = st.selectbox("Publication Day", days)
    
    # Submit button
    submitted = st.form_submit_button("See Result")

# Handle prediction when form is submitted
if submitted:
    if not time_length or genre is None or publication_day is None:
        st.error("Please enter the all required information.")
    else:
        try:
            features = {
                'time_length': time_length,
                'genre': genre,
                'publication_day': publication_day
            }
            
            if st.session_state.predictor.model is not None:
                prediction = st.session_state.predictor.predict(features)
                
                #TODO: frontend code - Results
                # Results
                st.markdown("## 📊 Result")
                st.markdown(f"Based on your podcast information, the average streaming time will be: **{prediction}**")
                
                # Get and display suggestions
                suggestions = get_suggestion(features)
                st.markdown("### 💡 Suggestion:")
                st.markdown(suggestions)
            else:
                st.error("Model not loaded. Please ensure the model is trained and saved correctly.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please make sure the time format is correct (HH:MM or MM:SS)")

# Add information about the model
st.markdown("""
---
### About the Model

This prediction model uses machine learning to analyze podcast metadata and predict listener behavior. 
The model takes into account:
- Episode length
- Genre
- Publication day

The predictions are based on patterns learned from historical podcast listening data.
""")
