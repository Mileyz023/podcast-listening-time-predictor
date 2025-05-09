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

def preprocess_user_input(features_dict):
    # 1. Parse time length MM:SS → float
    try:
        time_parts = features_dict['time_length'].strip().split(":")
        if len(time_parts) != 2 or not all(tp.isdigit() for tp in time_parts):
            raise ValueError("Invalid time format")
        minutes = int(time_parts[0])
        seconds = int(time_parts[1])
        episode_length = minutes * 60 + seconds
    except Exception:
        raise ValueError("Time Length must be in MM:SS format (e.g., 02:30)")

    # 2. Encode sentiment
    sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    sentiment = sentiment_map.get(features_dict['episode_sentiment'], 1)

    # 3. Manually build the feature vector
    base_dict = {
        'Episode_Length_minutes': episode_length,
        'Host_Popularity_percentage': 50.0,
        'Guest_Popularity_percentage': 50.0,
        'Number_of_Ads': 1,
        'Episode_Sentiment': sentiment
    }

    # 4. One-hot encode categorical features
    genres = ["Comedy", "News", "Education", "Business", "Technology", "Health", "Arts", "Sports", "Music", "Society"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    times = ["Morning", "Afternoon", "Evening", "Night"]

    for g in genres[1:]:  # drop_first=True
        base_dict[f'Genre_{g}'] = 1 if features_dict['genre'] == g else 0

    for d in days[1:]:
        base_dict[f'Publication_Day_{d}'] = 1 if features_dict['publication_day'] == d else 0

    for t in times[1:]:
        base_dict[f'Publication_Time_{t}'] = 1 if features_dict['publication_time'] == t else 0

    # 5. Convert to array and return
    return np.array([list(base_dict.values())])


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
    
    # Pre-trained model check
    model_path = f'pretrained_models/{selected_model.lower().replace(" ", "_")}.joblib'
    
    if not os.path.exists(model_path):
        st.info("Model not found. Training new model...")
        with st.spinner('Training model...'):
            train_df, test_df = load_data()
            X_train, y_train, X_eval, y_eval = preprocess_dataset(train_df)
            model = train_model(X_train, y_train, X_eval, y_eval, selected_model)

            #Ensure directory exists
            os.makedirs('pretrained_models', exist_ok=True)
            
            # Export trained model
            joblib.dump(model, model_path)
            st.success("Model trained successfully!")
    
    # Load the model
    model = joblib.load(model_path)
    
    # Make prediction
    try:
        #prediction = model.predict(pd.DataFrame([features]))
        input_array = preprocess_user_input(features)
        prediction = model.predict(input_array)

        predicted_minutes = float(prediction[0])  # extract scalar
        
        # Display results
        st.markdown("## 📊 Result")
        #st.markdown(f"Based on your podcast information, the average streaming time will be: **{prediction[0]:.2f}** minutes")
        st.markdown(f"Based on your podcast information, the average streaming time will be: **{predicted_minutes:.2f}** minutes")

        
        # Visualize the prediction
        st.markdown("### 📈 Visualization")
        
        # Convert time length to minutes
        time_parts = time_length.split(':')
        episode_minutes = int(time_parts[0]) * 60 + int(time_parts[1])
        
        # Create data for visualization
        data = {
            'Metric': ['Episode Length', 'Predicted Listening Time'],
            #'Minutes': [episode_minutes, prediction[0]]
            'Minutes': [episode_minutes, predicted_minutes]
        }
        df_viz = pd.DataFrame(data)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Metric', y='Minutes', data=df_viz, ax=ax)
        plt.title('Episode Length vs. Predicted Listening Time')
        plt.ylabel('Minutes')
        st.pyplot(fig)
        
        # Calculate completion rate
        #completion_rate = (prediction[0] / episode_minutes) * 100
        completion_rate = (predicted_minutes / episode_minutes) * 100
        st.markdown(f"**Completion Rate:** {completion_rate:.1f}%")
        
        # Display suggestions
        st.markdown("### 💡 Suggestions:")
        suggestions = []
        
        if completion_rate < 50:
            suggestions.append("Consider shortening your episode length")
        if publication_day in ['Saturday', 'Sunday']:
            suggestions.append("Consider publishing on weekdays for potentially better engagement")
        if publication_time == 'Night':
            suggestions.append("Consider publishing during morning or afternoon hours")
        if episode_sentiment == 'Negative':
            suggestions.append("Consider maintaining a more positive or neutral tone")
        
        if suggestions:
            for suggestion in suggestions:
                st.write(f"• {suggestion}")
        else:
            st.write("Your podcast parameters look optimal!")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please make sure the time format is correct (MM:SS)")

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