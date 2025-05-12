import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from utils import preprocess_dataset, load_data

st.set_page_config(page_title="Model Evaluation", layout="wide")
st.markdown("# Model Evaluation")

def report(y_true, y_pred, dataset_name):
    """Generate and display evaluation metrics for regression models"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'R²'],
        'Value': [mse, rmse, mae, r2]
    })
    
    st.markdown(f"### {dataset_name} Set Metrics")
    st.dataframe(metrics_df)
    
    return metrics_df

# Load data and models
@st.cache_data
def load_evaluation_data():
    train_df, test_df = load_data()
    X_train, y_train, X_eval, y_eval = preprocess_dataset(train_df, test_df)
    return X_train, y_train, X_eval, y_eval

# Load available models
@st.cache_resource(show_spinner=False)
def load_available_models():
    model_files = {
        "Linear Regression": 'pretrained_models/linear_regression.joblib',
        "Ridge Regression": 'pretrained_models/ridge_regression.joblib',
        "Polynomial Regression": 'pretrained_models/polynomial_regression.joblib'
    }
    
    available_models = {}
    for model_name, model_path in model_files.items():
        if os.path.exists(model_path):
            try:
                available_models[model_name] = joblib.load(model_path)
            except Exception as e:
                st.warning(f"Could not load {model_name}: {str(e)}")
    
    return available_models

try:
    # Load data
    X_train, y_train, X_eval, y_eval = load_evaluation_data()
    
    # Add a refresh button to force reload models
    if st.button("Refresh Available Models"):
        st.cache_resource.clear()
        st.rerun()
    
    # Load available models
    models = load_available_models()
    
    # Display model count for debugging
    st.write(f"Found {len(models)} trained models")
    
    if not models:
        st.error("No trained models found. Please train at least one model first.")
        st.info("You can train models on the 'Model Training and Inference' page.")
        st.stop()
    
    # Model selection
    selected_model = st.selectbox(
        "Select model to evaluate",
        list(models.keys())
    )
    
    model = models[selected_model]
    
    # Generate predictions
    y_train_pred = model.predict(X_train)
    y_eval_pred = model.predict(X_eval)
    
    # === Report ===
    st.markdown("## Report")
    report(y_train, y_train_pred, "TRAIN")
    report(y_eval, y_eval_pred, "TEST")

    st.markdown("### What does this mean?")
    st.markdown("MSE: Measures the average squared prediction error; lower values indicate better fit but are sensitive to large errors and scale-dependent.")
    st.markdown("RMSE: Represents the average prediction error in the original unit of measurement; easier to interpret practically, with lower values indicating better performance.")
    st.markdown("MAE: Reflects the average absolute prediction error; lower values suggest more accurate and robust performance, especially against outliers.")
    st.markdown("R²: Indicates that the model explains how much of the variance in the target data; higher values (closer to 1) mean better explanatory power.")
    
    # === Plotting ===
    st.markdown("## Plotting")
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter: Actual vs Predicted
    axs[0].scatter(y_eval_pred, y_eval, alpha=0.3)
    axs[0].plot([y_eval.min(), y_eval.max()], [y_eval.min(), y_eval.max()], 'r--', lw=2)
    axs[0].set_title('Actual vs Predicted (Test Set)')
    axs[0].set_xlabel('Predicted')
    axs[0].set_ylabel('Actual')
    axs[0].grid(True)
    
    # Histogram: Residuals
    residuals = y_eval - y_eval_pred
    axs[1].hist(residuals, bins=50, edgecolor='k', alpha=0.7)
    axs[1].set_title('Residuals Distribution (Test Set)')
    axs[1].set_xlabel('Residual')
    axs[1].set_ylabel('Frequency')
    axs[1].grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Additional evaluation metrics
    # st.markdown("## Additional Analysis")
    
    # Feature importance (for linear models)
    # if selected_model in ["Linear Regression", "Ridge Regression"]:
    #     st.markdown("### Feature Importance")
    #     try:
    #         # Get feature names from preprocessing
    #         feature_names = X_train.columns if hasattr(X_train, 'columns') else [f"Feature {i}" for i in range(X_train.shape[1])]
            
    #         # Get coefficients
    #         coeffs = model.W[1:] if hasattr(model, 'W') else model.coef_
            
    #         # Create DataFrame for visualization
    #         coef_df = pd.DataFrame({
    #             'Feature': feature_names,
    #             'Coefficient': coeffs.flatten()
    #         }).sort_values('Coefficient', ascending=False)
            
    #         # Plot feature importance
    #         fig, ax = plt.subplots(figsize=(10, 8))
    #         coef_df.plot(kind='barh', x='Feature', y='Coefficient', ax=ax)
    #         plt.title('Feature Importance')
    #         plt.xlabel('Coefficient Value')
    #         plt.tight_layout()
    #         st.pyplot(fig)
            
    #     except Exception as e:
    #         st.warning(f"Could not display feature importance: {str(e)}")
    
    # Residual analysis
    st.markdown("### Residual Analysis")
    
    # Create residual plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_eval_pred, residuals, alpha=0.5)
    ax.axhline(y=0, color='r', linestyle='-')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Plot')
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("### What does this mean?")
    st.markdown("The left scatter plot compares the actual target values (y-axis) with the model’s predicted values (x-axis) on the test set. The red dashed line represents perfect predictions. Ideally, points should align closely to the red dashed line.")
    st.markdown("The histogram on the right shows the distribution of residuals (Actual - Predicted) on the test set. Ideally, residuals should be centered around 0 and symmetrically distributed (like a bell curve), indicating unbiased predictions.")
    st.markdown("Residuals should ideally be normally distributed with a mean of 0. A large standard deviation indicates high variability in prediction errors. The min and max values provide insight into the range of errors.")

    
    
    # Residual statistics
    st.markdown("#### Residual Statistics")
    residual_stats = pd.DataFrame({
        'Metric': ['Mean', 'Std Dev', 'Min', 'Max'],
        'Value': [
            np.mean(residuals),
            np.std(residuals),
            np.min(residuals),
            np.max(residuals)
        ]
    })
    st.dataframe(residual_stats)


except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Make sure you have trained at least one model on the 'Model Training and Inference' page.")