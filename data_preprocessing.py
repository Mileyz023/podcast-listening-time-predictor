import pandas as pd
import numpy as np

def load_data():
    """Load training and testing datasets"""
    # Load training data
    train_data = pd.read_csv('data/train.csv')
    
    # Load testing data
    test_data = pd.read_csv('data/test.csv')
    
    return train_data, test_data

def preprocess_dataset(df, is_training=True, train_stats=None):
    """
    Preprocess the dataset
    
    Args:
        df (DataFrame): Raw dataset
        is_training (bool): Whether this is training data
        train_stats (dict): Statistics from training data for test set preprocessing
        
    Returns:
        DataFrame: Preprocessed dataset
        dict: Training statistics (if is_training=True)
    """
    # Make a copy
    df_cleaned = df.copy()
    
    if is_training:
        train_stats = {}
        # Calculate statistics from training data
        train_stats['episode_length_median'] = df_cleaned['Episode_Length_minutes'].median()
        train_stats['guest_popularity_median'] = df_cleaned['Guest_Popularity_percentage'].median()
        train_stats['number_of_ads_mode'] = df_cleaned['Number_of_Ads'].mode()[0]
    
    # 1. Missing Value Imputation
    df_cleaned['Episode_Length_minutes'] = df_cleaned['Episode_Length_minutes'].fillna(
        train_stats['episode_length_median'] if train_stats else df_cleaned['Episode_Length_minutes'].median()
    )
    df_cleaned['Guest_Popularity_percentage'] = df_cleaned['Guest_Popularity_percentage'].fillna(
        train_stats['guest_popularity_median'] if train_stats else df_cleaned['Guest_Popularity_percentage'].median()
    )
    df_cleaned['Number_of_Ads'] = df_cleaned['Number_of_Ads'].fillna(
        train_stats['number_of_ads_mode'] if train_stats else df_cleaned['Number_of_Ads'].mode()[0]
    )
    
    # 2. Clip Outliers
    df_cleaned['Episode_Length_minutes'] = np.clip(df_cleaned['Episode_Length_minutes'], None, 120)
    df_cleaned['Host_Popularity_percentage'] = np.clip(df_cleaned['Host_Popularity_percentage'], 0, 100)
    df_cleaned['Guest_Popularity_percentage'] = np.clip(df_cleaned['Guest_Popularity_percentage'], 0, 100)
    df_cleaned['Number_of_Ads'] = np.clip(df_cleaned['Number_of_Ads'], None, 3)
    
    if is_training:
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
    
    if is_training:
        return df_cleaned, train_stats
    else:
        return df_cleaned

def preprocess_data():
    """
    Main function to preprocess both training and test data
    
    Returns:
        tuple: (preprocessed_train_data, preprocessed_test_data)
    """
    # Load data
    train_data, test_data = load_data()
    
    # Preprocess training data
    train_data_cleaned, train_stats = preprocess_dataset(train_data, is_training=True)
    
    # Preprocess test data using training statistics
    test_data_cleaned = preprocess_dataset(test_data, is_training=False, train_stats=train_stats)
    
    # Ensure columns match between train and test
    train_cols = set(train_data_cleaned.columns)
    test_cols = set(test_data_cleaned.columns)
    
    # Add missing columns to test data
    for col in train_cols - test_cols:
        if col != 'Listening_Time_minutes':  # Don't add target column to test data
            test_data_cleaned[col] = 0
    
    # Remove extra columns from test data
    extra_cols = test_cols - train_cols
    test_data_cleaned = test_data_cleaned.drop(columns=list(extra_cols))
    
    # Ensure column order matches
    test_data_cleaned = test_data_cleaned[
        [col for col in train_data_cleaned.columns if col != 'Listening_Time_minutes']
    ]
    
    return train_data_cleaned, test_data_cleaned

if __name__ == "__main__":
    # Test the preprocessing pipeline
    train_data_cleaned, test_data_cleaned = preprocess_data()
    print("Training data shape:", train_data_cleaned.shape)
    print("Test data shape:", test_data_cleaned.shape) 