import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from utils import preprocess_dataset, load_data, train_model
from models.linear_regression_model import LinearRegression
from models.ridge_regression import RidgeRegression
from models.polynomial_regression import PolynomialRegression

st.set_page_config(page_title="Podcast Listening Time Predictor", layout="wide")
st.markdown("# Predict your podcast average streaming time")


# User Input
st.markdown("## 🎙️ Podcast Info")
with st.form("podcast_form"):
    episode_name = st.text_input("Episode Name", placeholder="Enter episode name")
    time_length = st.text_input("Time Length (MM:SS)", placeholder="02:00")
    genres = ["Comedy", "News", "Education", "Business", "Technology", "Health", "Arts", "Sports", "Music", "Society"]
    genre = st.selectbox("Genre", genres)
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    publication_day = st.selectbox("Publication Day", days)
    times = ["Morning", "Afternoon", "Evening", "Night"]
    publication_time = st.selectbox("Publication Time", times)
    sentiments = ["Positive", "Neutral", "Negative"]
    episode_sentiment = st.selectbox("Episode Sentiment", sentiments)
    available_models = ["Linear Regression", "Ridge Regression", "Polynomial Regression"]
    selected_model = st.selectbox("Select Model", available_models)

    submitted = st.form_submit_button("See Result")

if submitted:
    features = {
        'time_length': time_length,
        'genre': genre,
        'publication_day': publication_day,
        'publication_time': publication_time,
        'episode_sentiment': episode_sentiment
    }
    
    # # Pre-trained model check
    # model_path = f'models/{selected_model.lower().replace(" ", "_")}.joblib'
    
    # if not os.path.exists(model_path):
    #     st.info("Model not found. Training new model...")
    #     with st.spinner('Training model...'):
    #         train_df, test_df = load_data()
    #         X_train, y_train, X_eval, y_eval = preprocess_dataset(train_df)
    #         model = train_model(X_train, y_train, X_eval, y_eval, selected_model)
            
    #         # Export trained model
    #         joblib.dump(models[selected_model], model_path)
    #         st.success("Model trained successfully!")
    
    # # Load the model
    # model = joblib.load(model_path)
    
    # # Make prediction
    # try:
    #     prediction = model.predict(pd.DataFrame([features]))
        
    #     # Display results
    #     st.markdown("## 📊 Result")
    #     st.markdown(f"Based on your podcast information, the average streaming time will be: **{prediction[0]:.2f}** minutes")
        
    #     # Visualize the prediction
    #     st.markdown("### 📈 Visualization")
        
    #     # Convert time length to minutes
    #     time_parts = time_length.split(':')
    #     episode_minutes = int(time_parts[0]) * 60 + int(time_parts[1])
        
    #     # Create data for visualization
    #     data = {
    #         'Metric': ['Episode Length', 'Predicted Listening Time'],
    #         'Minutes': [episode_minutes, prediction[0]]
    #     }
    #     df_viz = pd.DataFrame(data)
        
    #     # Create bar chart
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #     sns.barplot(x='Metric', y='Minutes', data=df_viz, ax=ax)
    #     plt.title('Episode Length vs. Predicted Listening Time')
    #     plt.ylabel('Minutes')
    #     st.pyplot(fig)
        
    #     # Calculate completion rate
    #     completion_rate = (prediction[0] / episode_minutes) * 100
    #     st.markdown(f"**Completion Rate:** {completion_rate:.1f}%")
        
    #     # Display suggestions
    #     st.markdown("### 💡 Suggestions:")
    #     suggestions = []
        
    #     if completion_rate < 50:
    #         suggestions.append("Consider shortening your episode length")
    #     if publication_day in ['Saturday', 'Sunday']:
    #         suggestions.append("Consider publishing on weekdays for potentially better engagement")
    #     if publication_time == 'Night':
    #         suggestions.append("Consider publishing during morning or afternoon hours")
    #     if episode_sentiment == 'Negative':
    #         suggestions.append("Consider maintaining a more positive or neutral tone")
        
    #     if suggestions:
    #         for suggestion in suggestions:
    #             st.write(f"• {suggestion}")
    #     else:
    #         st.write("Your podcast parameters look optimal!")
            
    # except Exception as e:
    #     st.error(f"An error occurred: {str(e)}")
    #     st.error("Please make sure the time format is correct (MM:SS)")

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