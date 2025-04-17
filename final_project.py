import streamlit as st                  # pip install streamlit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model.podcast_predictor import PodcastPredictor, get_suggestion
import os
from helper_functions import fetch_dataset

st.set_page_config(page_title="Podcast Listening Time Predictor", layout="wide")

st.markdown("# Podcast Listening Time Predictor")

st.markdown("""
This application helps podcast creators predict the average listening time for their episodes 
based on various metadata factors. Simply input your episode details below to get a prediction.
""")

# Initialize the predictor in session state if it doesn't exist
if 'predictor' not in st.session_state:
    st.session_state.predictor = PodcastPredictor()

    model_path = 'model/podcast_model.joblib'
    if os.path.exists(model_path):
        st.session_state.predictor.load_model(model_path)
        st.success("Model loaded successfully!")
    else:
        st.warning("Model not found. Please train the model first.")

# Create the input form
st.markdown("## 🎙️ Podcast Info")

with st.form("podcast_form"):
    # Episode name (for user reference only)
    episode_name = st.text_input("Episode Name", placeholder="Enter episode name")
    
    # Time Length
    time_length = st.text_input("Time Length (MM:SS)", placeholder="02:00")
    
    # Genre selection
    genres = ["Comedy", "News", "Education", "Business", "Technology", "Health", "Arts", "Sports", "Music", "Society"]
    genre = st.selectbox("Genre", genres)
    
    # Publication day
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    publication_day = st.selectbox("Publication Day", days)
    
    # Publication time
    times = ["Morning", "Afternoon", "Evening", "Night"]
    publication_time = st.selectbox("Publication Time", times)
    
    # Episode sentiment
    sentiments = ["Positive", "Neutral", "Negative"]
    episode_sentiment = st.selectbox("Episode Sentiment", sentiments)
    
    # Submit button
    submitted = st.form_submit_button("See Result")

# Handle prediction when form is submitted
if submitted:
    if not time_length or genre is None or publication_day is None:
        st.error("Please enter all required information.")
    else:
        try:
            # Create features dictionary
            features = {
                'time_length': time_length,
                'genre': genre,
                'publication_day': publication_day,
                'publication_time': publication_time,
                'episode_sentiment': episode_sentiment
            }
            
            if st.session_state.predictor.model is not None:
                # Make prediction
                prediction = st.session_state.predictor.predict(features)
                
                # Display results
                st.markdown("## 📊 Result")
                st.markdown(f"Based on your podcast information, the average streaming time will be: **{prediction}**")
                
                # Get and display suggestions
                suggestions = get_suggestion(features)
                st.markdown("### 💡 Suggestions:")
                st.markdown(suggestions)
                
                # Visualize the prediction
                st.markdown("### 📈 Visualization")
                
                # Create a bar chart comparing episode length and predicted listening time
                time_parts = time_length.split(':')
                if len(time_parts) == 2:
                    episode_minutes = int(time_parts[0]) * 60 + int(time_parts[1])
                else:
                    episode_minutes = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
                
                # Convert prediction to minutes
                pred_parts = prediction.split(':')
                pred_minutes = int(pred_parts[0]) * 60 + int(pred_parts[1])
                
                # Create data for visualization
                data = {
                    'Metric': ['Episode Length', 'Predicted Listening Time'],
                    'Minutes': [episode_minutes, pred_minutes]
                }
                df_viz = pd.DataFrame(data)
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Metric', y='Minutes', data=df_viz, ax=ax)
                plt.title('Episode Length vs. Predicted Listening Time')
                plt.ylabel('Minutes')
                st.pyplot(fig)
                
                # Calculate completion rate
                completion_rate = (pred_minutes / episode_minutes) * 100
                st.markdown(f"**Completion Rate:** {completion_rate:.1f}%")
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
- Publication time
- Episode sentiment

The predictions are based on patterns learned from historical podcast listening data.
""")
