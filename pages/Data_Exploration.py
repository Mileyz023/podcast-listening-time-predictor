import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Dataset Exploration", layout="wide")

st.markdown("# Podcast Dataset Exploration")

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        df = pd.read_csv('data/train.csv')
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Load the dataset
df = load_data()

if df is not None:
    # Basic Dataset Information
    st.markdown("## Dataset Overview")
    st.write(f"Number of records: {df.shape[0]}")
    st.write(f"Number of features: {df.shape[1]}")
    
    # Display sample data
    st.markdown("### Sample Data")
    st.dataframe(df.head())
    
    # Missing Values Analysis
    st.markdown("### Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values)
    
    # Basic Statistics
    st.markdown("### Basic Statistics")
    st.write(df.describe())
    
    # Numerical Features Analysis
    st.markdown("## Numerical Features Analysis")
    
    # Target Variable Distribution
    st.markdown("### Target Variable: Listening Time")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Histogram
    sns.histplot(df['Listening_Time_minutes'], kde=True, ax=ax1)
    ax1.set_title("Distribution of Listening Time")
    ax1.set_xlabel("Listening Time (minutes)")
    
    # Boxplot
    sns.boxplot(x=df['Listening_Time_minutes'], ax=ax2)
    ax2.set_title("Boxplot of Listening Time")
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Other Numerical Features
    st.markdown("### Other Numerical Features")
    numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage',
                     'Guest_Popularity_percentage', 'Number_of_Ads']
    
    # Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f"Distribution of {col}")
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Correlation Analysis
    st.markdown("### Correlation Analysis")
    
    # Add target variable to correlation analysis
    corr_cols = numerical_cols + ['Listening_Time_minutes']
    corr_matrix = df[corr_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    plt.title("Correlation Heatmap")
    st.pyplot(fig)
    
    # Categorical Features Analysis
    st.markdown("## Categorical Features Analysis")
    categorical_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
    
    for col in categorical_cols:
        st.markdown(f"### {col} Analysis")
        
        # Create two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Distribution of categories
        sns.countplot(data=df, x=col, ax=ax1)
        ax1.set_title(f"Distribution of {col}")
        ax1.tick_params(axis='x', rotation=45)
        
        # Average listening time by category
        sns.barplot(data=df, x=col, y='Listening_Time_minutes', ax=ax2)
        ax2.set_title(f"Average Listening Time by {col}")
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Scatter Plots
    st.markdown("## Relationship with Target Variable")
    
    for col in numerical_cols:
        st.markdown(f"### {col} vs Listening Time")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x=col, y='Listening_Time_minutes', alpha=0.5)
        plt.title(f"{col} vs Listening Time")
        st.pyplot(fig)
    
    # Additional Insights
    st.markdown("## Additional Insights")
    
    # Podcast Names Analysis
    st.markdown("### Podcast Analysis")
    podcast_counts = df['Podcast_Name'].value_counts()
    st.write(f"Number of unique podcasts: {len(podcast_counts)}")
    
    # Top 10 podcasts by number of episodes
    st.markdown("#### Top 10 Podcasts by Number of Episodes")
    st.write(podcast_counts.head(10))
    
    # Top 10 podcasts by average listening time
    st.markdown("#### Top 10 Podcasts by Average Listening Time")
    avg_listening_time = df.groupby('Podcast_Name')['Listening_Time_minutes'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    avg_listening_time = avg_listening_time[avg_listening_time['count'] >= 5]  # Filter for podcasts with at least 5 episodes
    st.write(avg_listening_time.head(10))
    
    # Time-based Analysis
    st.markdown("### Time-based Analysis")
    
    # Average listening time by day and time
    pivot_table = df.pivot_table(
        values='Listening_Time_minutes',
        index='Publication_Day',
        columns='Publication_Time',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd')
    plt.title("Average Listening Time by Day and Time")
    st.pyplot(fig)
    
else:
    st.error("Please make sure the dataset is available in the data directory.") 