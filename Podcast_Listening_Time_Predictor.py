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
    # parse time length MM:SS → float
    try:
        time_parts = features_dict['time_length'].strip().split(":")
        if len(time_parts) != 2 or not all(tp.isdigit() for tp in time_parts):
            raise ValueError("Invalid time format")
        minutes = int(time_parts[0])
        seconds = int(time_parts[1])
        # Convert to total minutes
        episode_length = minutes + seconds/60
    except Exception:
        raise ValueError("Time Length must be in MM:SS format (e.g., 02:30)")

    # Encode sentiment
    sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    sentiment = sentiment_map.get(features_dict['episode_sentiment'], 1)

    # Get user-provided values
    host_popularity = float(features_dict.get('host_popularity', 50.0))
    guest_popularity = float(features_dict.get('guest_popularity', 50.0))
    number_of_ads = int(features_dict.get('number_of_ads', 1))

    # Create feature list in the exact order the model expects
    features = [
        episode_length,
        host_popularity,
        guest_popularity,
        number_of_ads,
        sentiment
    ]
    
    # One-hot encode categorical features with drop_first=True
    # This means we omit the first category for each group
    genres = ["Comedy", "News", "Education", "Business", "Technology", "Health", "Arts", "Sports", "Music", "Society"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    times = ["Morning", "Afternoon", "Evening", "Night"]

    # Add genre features (omit first category - Comedy)
    for g in genres[1:]:
        features.append(1 if features_dict['genre'] == g else 0)
    
    # Add day features (omit first category - Monday)
    for d in days[1:]:
        features.append(1 if features_dict['publication_day'] == d else 0)
    
    # Add time features (omit first category - Morning)
    for t in times[1:]:
        features.append(1 if features_dict['publication_time'] == t else 0)

    # Convert to array and return
    return np.array([features])


st.set_page_config(page_title="Podcast Listening Time Predictor", layout="wide")
st.markdown("# Podcast Listening Time Predictor")
st.markdown("""
This tool predicts how long listeners will engage with your podcast based on various factors.
Enter your podcast details below to get a prediction.
""")

# User Input
st.markdown("## 🎙️ Podcast Info")
with st.form("podcast_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        episode_name = st.text_input("Episode Name", placeholder="Enter episode name")
        time_length = st.text_input("Time Length (MM:SS)", placeholder="45:00")
        host_popularity = st.slider("Host Popularity (%)", 0, 100, 50)
        guest_popularity = st.slider("Guest Popularity (%)", 0, 100, 50)
        number_of_ads = st.number_input("Number of Ads", min_value=0, max_value=10, value=2)
    
    with col2:
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

    submitted = st.form_submit_button("Predict Listening Time")

if submitted:
    # Set default values if fields are empty
    if not episode_name:
        episode_name = "Unnamed Podcast"
    
    if not time_length:
        time_length = "45:00"
        
    features = {
        'time_length': time_length,
        'genre': genre,
        'publication_day': publication_day,
        'publication_time': publication_time,
        'episode_sentiment': episode_sentiment,
        'host_popularity': host_popularity,
        'guest_popularity': guest_popularity,
        'number_of_ads': number_of_ads
    }
    
    # Pre-trained model check
    model_path = f'pretrained_models/{selected_model.lower().replace(" ", "_")}.joblib'
    
    if not os.path.exists(model_path):
        st.info("Model not found. Training new model...")
        with st.spinner('Training model...'):
            train_df, test_df = load_data()
            X_train, y_train, X_eval, y_eval = preprocess_dataset(train_df)
            model = train_model(X_train, y_train, X_eval, y_eval, selected_model)

            # Ensure directory exists
            os.makedirs('pretrained_models', exist_ok=True)
            
            # Export trained model
            joblib.dump(model, model_path)
            st.success("Model trained successfully!")
    
    # Load the model
    model = joblib.load(model_path)
    
    # Make prediction
    try:
        input_array = preprocess_user_input(features)
        prediction = model.predict(input_array)

        predicted_minutes = float(prediction[0])  # extract scalar
        
        # Ensure prediction is not negative
        predicted_minutes = max(0, predicted_minutes)
        
        # Display results
        st.markdown("## 📊 Prediction Results")
        
        # Convert time length to minutes for comparison
        time_parts = time_length.split(':')
        episode_minutes = int(time_parts[0]) + int(time_parts[1])/60
        
        # Calculate completion rate
        completion_rate = (predicted_minutes / episode_minutes) * 100
        completion_rate = min(100, completion_rate)  # Cap at 100%
        
        # Create columns for metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("Episode Length", f"{episode_minutes:.1f} min")
        
        with metric_col2:
            st.metric("Predicted Listening Time", f"{predicted_minutes:.1f} min")
            
        with metric_col3:
            st.metric("Completion Rate", f"{completion_rate:.1f}%")
        
        # Visualize the prediction
        st.markdown("### 📈 Visualization")
        
        # Create data for visualization
        data = {
            'Metric': ['Episode Length', 'Predicted Listening Time'],
            'Minutes': [episode_minutes, predicted_minutes]
        }
        df_viz = pd.DataFrame(data)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = sns.barplot(x='Metric', y='Minutes', data=df_viz, ax=ax, palette=['#1f77b4', '#ff7f0e'])
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars.patches):
            bars.text(
                bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 0.3,
                f"{df_viz['Minutes'].iloc[i]:.1f}",
                ha='center', 
                fontsize=11
            )
            
        plt.title('Episode Length vs. Predicted Listening Time')
        plt.ylabel('Minutes')
        st.pyplot(fig)
        
       
        
        # Display suggestions
        st.markdown("### 💡 Suggestions to Improve Engagement:")
        suggestions = []
        
        if completion_rate < 60:
            suggestions.append("Consider shortening your episode length for better listener retention")
        
        if number_of_ads > 3:
            suggestions.append("Consider reducing the number of ads to improve listener experience")
            
        if publication_day in ['Saturday', 'Sunday'] and completion_rate < 70:
            suggestions.append("Consider publishing on weekdays for potentially better engagement")
            
        if publication_time == 'Night' and completion_rate < 70:
            suggestions.append("Consider publishing during morning or afternoon hours")
            
        if episode_sentiment == 'Negative' and completion_rate < 80:
            suggestions.append("Consider maintaining a more positive or neutral tone")
            
        if host_popularity < 40 or guest_popularity < 40:
            suggestions.append("Consider featuring more popular hosts or guests to increase engagement")
        
        if suggestions:
            for suggestion in suggestions:
                st.write(f"• {suggestion}")
        else:
            st.write("• Your podcast parameters look optimal! Expected listener engagement is high.")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please make sure all inputs are valid and the time format is correct (MM:SS)")

# Add information about the model
st.markdown("""
---
### About the Prediction Model

This prediction model uses machine learning to analyze podcast metadata and predict listener behavior. 
The model takes into account:

- **Episode length**: Total duration of the podcast
- **Host & Guest Popularity**: Perceived popularity percentages
- **Number of Ads**: How many advertisements are included
- **Genre**: The podcast's primary category
- **Publication Day & Time**: When the episode is released
- **Episode Sentiment**: Overall emotional tone of the content

The predictions are based on patterns learned from historical podcast listening data. Different models may provide varying predictions based on their underlying algorithms.
""")