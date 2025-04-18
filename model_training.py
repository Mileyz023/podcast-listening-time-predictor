import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.net(x)

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name=""):
    """
    Evaluate model performance and create visualizations
    """
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = np.mean((y_train - y_train_pred) ** 2)
    test_mse = np.mean((y_test - y_test_pred) ** 2)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_r2 = 1 - np.sum((y_train - y_train_pred) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
    test_r2 = 1 - np.sum((y_test - y_test_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    
    print(f"\n{model_name} Results:")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Actual vs Predicted plot
    ax1.scatter(y_test, y_test_pred, alpha=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.set_title(f'{model_name}: Actual vs Predicted')
    
    # Residuals plot
    residuals = y_test - y_test_pred
    ax2.hist(residuals, bins=30, alpha=0.5)
    ax2.set_xlabel('Residual')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{model_name}: Residuals Distribution')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2
    }

def train_neural_network(X_train, y_train, X_test, y_test):
    """
    Train a neural network model
    """
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create model
    model = NeuralNetwork(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 100
    batch_size = 32
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
    
    # Create a wrapper class for sklearn compatibility
    class NNWrapper:
        def __init__(self, model):
            self.model = model
        
        def predict(self, X):
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                return self.model(X_tensor).numpy()
    
    return NNWrapper(model)

def train_and_evaluate_models(train_data, test_data):
    """
    Train and evaluate multiple models
    
    Args:
        train_data (DataFrame): Preprocessed training data
        test_data (DataFrame): Preprocessed test data
        
    Returns:
        dict: Trained models
    """
    # Prepare data
    X = train_data.drop(['id', 'Listening_Time_minutes'], axis=1)
    y = train_data['Listening_Time_minutes']
    
    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Polynomial Regression': Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('linear', LinearRegression())
        ]),
        'Neural Network': None  # Will be trained separately
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        if name != 'Neural Network':
            print(f"\nTraining {name}...")
            model.fit(X_train_scaled, y_train)
            results[name] = evaluate_model(model, X_train_scaled, y_train, X_val_scaled, y_val, name)
    
    # Train Neural Network
    print("\nTraining Neural Network...")
    nn_model = train_neural_network(X_train_scaled, y_train, X_val_scaled, y_val)
    results['Neural Network'] = evaluate_model(nn_model, X_train_scaled, y_train, X_val_scaled, y_val, "Neural Network")
    models['Neural Network'] = nn_model
    
    # Compare model performances
    performance_df = pd.DataFrame(results).T
    print("\nModel Performance Comparison:")
    print(performance_df)
    
    # Visualize model comparison
    plt.figure(figsize=(10, 6))
    performance_df[['train_rmse', 'test_rmse']].plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return models

if __name__ == "__main__":
    from data_preprocessing import preprocess_data
    
    # Test the training pipeline
    train_data, test_data = preprocess_data()
    models = train_and_evaluate_models(train_data, test_data) 