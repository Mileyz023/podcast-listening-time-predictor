import pandas as pd
import numpy as np

#TODO: remove all scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

class PodcastPredictor:  #TODO: Input Model
    def __init__(self):
        self.model = None
        self.preprocessor = None
        
    def preprocess_features(self, X):
        """
        Preprocess features for prediction
        
        Args:
            X (DataFrame): Features to preprocess
            
        Returns:
            DataFrame: Preprocessed features
        """
        # Convert time length to minutes
        def time_to_minutes(time_str):
            try:
                parts = time_str.split(':')
                if len(parts) == 2:
                    minutes = int(parts[0]) * 1 + int(parts[1]) / 60
                else:
                    minutes = int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
                return round(minutes, 2)
            except:
                return None

        # Create preprocessing steps for different types of features
        numeric_features = []
        categorical_features = ['genre', 'publication_day']
        
        # Create preprocessing pipelines
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', sparse=False)
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return X
    
    def preprocess_dataset(self, df):
        """
        Preprocess the entire dataset for training
        
        Args:
            df (DataFrame): Raw dataset
            
        Returns:
            DataFrame: Preprocessed dataset
        """
        # Make a copy
        df_cleaned = df.copy()
        
        # 1. Missing Value Imputation
        df_cleaned['Episode_Length_minutes'] = df_cleaned['Episode_Length_minutes'].fillna(df_cleaned['Episode_Length_minutes'].median())
        df_cleaned['Guest_Popularity_percentage'] = df_cleaned['Guest_Popularity_percentage'].fillna(df_cleaned['Guest_Popularity_percentage'].median())
        df_cleaned['Number_of_Ads'] = df_cleaned['Number_of_Ads'].fillna(df_cleaned['Number_of_Ads'].mode()[0])
        
        # 2. Clip Outliers
        df_cleaned['Episode_Length_minutes'] = np.clip(df_cleaned['Episode_Length_minutes'], None, 120)
        df_cleaned['Host_Popularity_percentage'] = np.clip(df_cleaned['Host_Popularity_percentage'], 0, 100)
        df_cleaned['Guest_Popularity_percentage'] = np.clip(df_cleaned['Guest_Popularity_percentage'], 0, 100)
        df_cleaned['Number_of_Ads'] = np.clip(df_cleaned['Number_of_Ads'], None, 3)
        
        # 3. Correct Overflow in Target
        df_cleaned['Listening_Time_minutes'] = np.where(
            df_cleaned['Listening_Time_minutes'] > df_cleaned['Episode_Length_minutes'],
            df_cleaned['Episode_Length_minutes'],
            df_cleaned['Listening_Time_minutes']
        )
        
        # 4. Extract numeric part from Episode_Title
        df_cleaned['Episode_Title_Numeric'] = df_cleaned['Episode_Title'].str.extract(r'(\d+)').astype(int)
        
        # 5. Drop high-cardinality or unused features
        df_cleaned.drop(columns=['Podcast_Name', 'Episode_Title'], inplace=True)
        
        # 6. Ordinal encoding for Episode_Sentiment
        sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
        df_cleaned['Episode_Sentiment'] = df_cleaned['Episode_Sentiment'].map(sentiment_map)
        
        # 7. One-hot encode nominal categorical features
        df_cleaned = pd.get_dummies(
            df_cleaned,
            columns=['Genre', 'Publication_Day', 'Publication_Time'],
            drop_first=True  # avoids multicollinearity
        )
        
        # 8. Convert boolean columns to integers
        bool_cols = df_cleaned.select_dtypes(include='bool').columns
        df_cleaned[bool_cols] = df_cleaned[bool_cols].astype(int)
        
        return df_cleaned

    def train_model(self, train_data):
        """
        Train the model on the preprocessed dataset
        
        Args:
            train_data (DataFrame): Preprocessed training data
            
        Returns:
            object: Trained model
        """
        # Prepare features and target
        X = train_data.drop(columns=['id', 'Listening_Time_minutes'])
        y = train_data['Listening_Time_minutes']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create the model pipeline
        self.model = Pipeline([
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        return self.model

    def predict(self, features):
        """
        Predict average listening time for a podcast episode
        
        Args:
            features (dict): Dictionary containing:
                - time_length (str): Episode length in format "HH:MM" or "MM:SS"
                - genre (str): Podcast genre
                - publication_day (str): Day of publication
                
        Returns:
            float: Predicted average listening time in minutes
        """
        # Convert input features to DataFrame
        input_df = pd.DataFrame([features])
        
        # Make prediction
        prediction = self.model.predict(input_df)[0]
        
        # Convert prediction to time format (HH:MM)
        hours = int(prediction // 60)
        minutes = int(prediction % 60)
        
        return f"{hours:02d}:{minutes:02d}"

    def save_model(self, filepath):
        """Save the trained model to a file"""
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        """Load a trained model from a file"""
        self.model = joblib.load(filepath)


def get_suggestion(features): #TODO: diplsy suggestions
    """
    Generate suggestions based on the podcast metadata
    
    Args:
        features (dict): Podcast features including genre, time_length, etc.
    
    Returns:
        str: Suggestion string
    """
    suggestions = []
    
    # Time-based suggestions
    time_parts = features['time_length'].split(':')
    minutes = int(time_parts[0])
    
    if minutes > 60:
        suggestions.append("Shorten intro")
    
    # Day-based suggestions
    if features['publication_day'].lower() not in ['wednesday', 'thursday']:
        suggestions.append("Try Wednesday release")
    
    # Add promotion suggestion
    suggestions.append("Boost promotion")
    
    return " · ".join(suggestions) 