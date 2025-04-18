import pandas as pd
import numpy as np
import sys
import os
from models.linear_regression_model import LinearRegression
from models.ridge_regression import RidgeRegression
from models.polynomial_regression import PolynomialRegression


def load_data():
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    
    return train_data, test_data

def preprocess_dataset(train_df, is_training=True, train_stats=None):
    # 1. Missing Value Imputation
    train_df['Episode_Length_minutes'] = train_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].median())
    train_df['Guest_Popularity_percentage'] = train_df['Guest_Popularity_percentage'].fillna(train_df['Guest_Popularity_percentage'].median())
    train_df['Number_of_Ads'] = train_df['Number_of_Ads'].fillna(train_df['Number_of_Ads'].mode()[0])

    # 2. Clip Outliers
    train_df['Episode_Length_minutes'] = np.clip(train_df['Episode_Length_minutes'], None, 120)
    train_df['Host_Popularity_percentage'] = np.clip(train_df['Host_Popularity_percentage'], 0, 100)
    train_df['Guest_Popularity_percentage'] = np.clip(train_df['Guest_Popularity_percentage'], 0, 100)
    train_df['Number_of_Ads'] = np.clip(train_df['Number_of_Ads'], None, 3)

    # 3. Correct Overflow in Target
    train_df['Listening_Time_minutes'] = np.where(
        train_df['Listening_Time_minutes'] > train_df['Episode_Length_minutes'],
        train_df['Episode_Length_minutes'],
        train_df['Listening_Time_minutes']
    )

    # Extract numeric part from Episode_Title
    train_df['Episode_Title_Numeric'] = train_df['Episode_Title'].str.extract(r'(\d+)').astype(int)

    # Drop original string column and high-cardinality Podcast_Name
    train_df.drop(columns=['Episode_Title', 'Podcast_Name'], inplace=True)

    # Ordinal encoding for Episode_Sentiment
    sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    train_df['Episode_Sentiment'] = train_df['Episode_Sentiment'].map(sentiment_map)

    # One-hot encode nominal categorical features
    train_df = pd.get_dummies(
        train_df,
        columns=['Genre', 'Publication_Day', 'Publication_Time'],
        drop_first=True  # avoids multicollinearity
    )

    bool_cols = train_df.select_dtypes(include='bool').columns
    train_df[bool_cols] = train_df[bool_cols].astype(int)

    # 1. Drop 'id' and 'Episode_Title_Numeric' from the dataframe
    df_filtered = train_df.drop(columns=['id', 'Episode_Title_Numeric'])

    # 2. Shuffle indices
    np.random.seed(50)
    shuffled_indices = np.random.permutation(len(df_filtered))
    split_index = int(len(df_filtered) * 0.8)
    train_indices = shuffled_indices[:split_index]
    eval_indices = shuffled_indices[split_index:]

    # 3. Features and target
    feature_columns = [col for col in df_filtered.columns if col != 'Listening_Time_minutes']
    X_all = df_filtered[feature_columns].values
    Y_all = df_filtered[['Listening_Time_minutes']].values

    # 4. Split manually
    X_train = X_all[train_indices]
    y_train = Y_all[train_indices]
    X_eval = X_all[eval_indices]
    y_eval = Y_all[eval_indices]

    return X_train, y_train, X_eval, y_eval

def train_model(X_train, y_train, X_eval, y_eval, model_name):
    print("start training!")
    if model_name == "Linear Regression":
        model = LinearRegression(learning_rate=0.01, num_iterations=100)
    elif model_name == "Ridge Regression":
        model = RidgeRegression(learning_rate=0.01, num_iterations=100, l2_penalty=1000)
    elif model_name == "Polynomial Regression":
        model = PolynomialRegression(degree=2, learning_rate=0.01, num_iterations=100)
    else:
        raise ValueError(f"Model {model_name} not supported")

    model.fit(X_train, y_train)
    return model

def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - ss_residual / ss_total

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def max_error(y_true, y_pred):
    return np.max(np.abs(y_true - y_pred))

def mean_error(y_true, y_pred):
    return np.mean(y_true - y_pred)  # bias direction

def evaluate_model(model, X_train, y_train, X_eval, y_eval):
    y_train_pred = model.predict(X_train)
    y_eval_pred = model.predict(X_eval)

def report(y_true, y_pred, label):
    print(f"{label} SET:")
    print(f"  MSE         = {mean_squared_error(y_true, y_pred):.4f}")
    print(f"  RMSE        = {root_mean_squared_error(y_true, y_pred):.4f}")
    print(f"  MAE         = {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"  R²          = {r2_score(y_true, y_pred):.4f}")
    print(f"  Max Error   = {max_error(y_true, y_pred):.4f}")
    print(f"  Mean Error  = {mean_error(y_true, y_pred):.4f}")
    print()

# # === Report ===
# report(y_train, y_train_pred, "TRAIN")
# report(y_eval, y_eval_pred, "TEST")

# # === Plotting ===
# import matplotlib.pyplot as plt
# fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# # Scatter: Actual vs Predicted
# axs[0].scatter(y_eval, y_eval_pred, alpha=0.3)
# axs[0].plot([y_eval.min(), y_eval.max()], [y_eval.min(), y_eval.max()], 'r--', lw=2)
# axs[0].set_title('Actual vs Predicted (Test Set)')
# axs[0].set_xlabel('Actual')
# axs[0].set_ylabel('Predicted')
# axs[0].grid(True)

# # Histogram: Residuals
# residuals = y_eval - y_eval_pred
# axs[1].hist(residuals, bins=50, edgecolor='k', alpha=0.7)
# axs[1].set_title('Residuals Distribution (Test Set)')
# axs[1].set_xlabel('Residual')
# axs[1].set_ylabel('Frequency')
# axs[1].grid(True)

# plt.tight_layout()
# plt.show()
