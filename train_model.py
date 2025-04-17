import pandas as pd
import numpy as np
from model.podcast_predictor import PodcastPredictor
import os

def generate_sample_data(n_samples=1000):
    """Generate sample data for training the model"""
    np.random.seed(42)
    
    # Generate random features
    genres = ["Comedy", "News", "Education", "Business", "Technology", 
             "Health", "Arts", "Sports", "Music", "Society"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", 
            "Friday", "Saturday", "Sunday"]
    
    data = {
        'time_length': [f"{np.random.randint(0, 3):02d}:{np.random.randint(0, 60):02d}" 
                       for _ in range(n_samples)],
        'genre': np.random.choice(genres, n_samples),
        'publication_day': np.random.choice(days, n_samples)
    }
    
    # Generate target variable (average_listening_time) with some patterns
    df = pd.DataFrame(data)
    
    # Convert time_length to minutes for calculation
    minutes = df['time_length'].apply(lambda x: 
        int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
    
    # Generate listening time with some patterns:
    # - Shorter episodes tend to be listened to more completely
    # - Wednesday releases tend to get more listening time
    # - Certain genres tend to perform better
    base_listening = minutes * (0.8 + np.random.normal(0, 0.1, n_samples))
    
    # Day effect
    day_effect = df['publication_day'].map({
        'Wednesday': 1.2,
        'Thursday': 1.1,
        'Tuesday': 1.0,
        'Monday': 0.9,
        'Friday': 0.9,
        'Saturday': 0.8,
        'Sunday': 0.8
    })
    
    # Genre effect
    genre_effect = df['genre'].map({
        'Comedy': 1.2,
        'News': 1.1,
        'Education': 1.0,
        'Business': 1.0,
        'Technology': 1.1,
        'Health': 0.9,
        'Arts': 0.9,
        'Sports': 1.0,
        'Music': 0.8,
        'Society': 0.9
    })
    
    # Combine effects
    listening_time = base_listening * day_effect * genre_effect
    
    # Add some noise
    listening_time = listening_time * (1 + np.random.normal(0, 0.1, n_samples))
    
    # Ensure non-negative values and convert to int
    data['average_listening_time'] = np.maximum(0, listening_time).astype(int)
    
    return pd.DataFrame(data)

def main():
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Generate sample training data
    print("Generating sample training data...")
    train_data = generate_sample_data()
    
    # Initialize and train the model
    print("Training model...")
    predictor = PodcastPredictor()
    predictor.train_model(train_data)
    
    # Save the model
    print("Saving model...")
    predictor.save_model('model/podcast_model.joblib')
    
    print("Done! The model has been trained and saved.")

if __name__ == "__main__":
    main() 