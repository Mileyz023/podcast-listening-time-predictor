# Podcast Listening Time Predictor

## Overview

This application predicts how long listeners will engage with a podcast based on various podcast characteristics. Using machine learning models, it analyzes factors such as episode length, genre, publication timing, and content sentiment to estimate average listening duration.

## Features

- **Prediction Engine**: Uses three different regression models (Linear, Ridge, and Polynomial) to predict listening time
- **Interactive Interface**: Simple form-based input for podcast details
- **Visual Results**: Graphical representation of predicted listening time vs. episode length
- **Engagement Insights**: Provides completion rate metrics and suggestions to improve listener engagement
- **User-Friendly**: Intuitive design with clear explanations and error handling

## Models

The application employs three different regression models:

1. **Linear Regression**: Basic model that assumes a linear relationship between features and listening time
2. **Ridge Regression**: Regularized linear model that helps prevent overfitting
3. **Polynomial Regression**: Captures non-linear relationships between podcast features and listening time

## How to Use

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   streamlit run Podcast_Listening_Time_Predictor.py
   ```
4. Enter your podcast details in the form:
   - Episode name and length
   - Host and guest popularity ratings
   - Number of advertisements
   - Genre, publication day and time
   - Overall episode sentiment
5. Select a prediction model
6. Click "Predict Listening Time" to see results

## Input Features

- **Episode Length**: Duration in MM:SS format
- **Host/Guest Popularity**: Percentage ratings (0-100%)
- **Number of Ads**: Count of advertisements in the episode
- **Genre**: Category of podcast content
- **Publication Day/Time**: When the episode is released
- **Episode Sentiment**: Overall emotional tone (Positive, Neutral, Negative)

## Results

The application provides:
- Predicted listening time in minutes
- Completion rate percentage
- Visual comparison between episode length and predicted listening time
- Personalized suggestions to improve listener engagement

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib


## Future Improvements

- Add more advanced models like Random Forest or Gradient Boosting
- Implement feature importance visualization
- Add time-series analysis for seasonal trends
- Enable batch prediction for multiple episodes
- Add user accounts to track prediction history