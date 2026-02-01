import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import config
import os
def run_eda():
    print("Running Exploratory Data Analysis...")
    if not os.path.exists(config.CLEANED_DATA_PATH):
        print("Cleaned data not found. Please run data_ingestion.py first.")
        return
    df = pd.read_csv(config.CLEANED_DATA_PATH)
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='planet_type', data=df, order=df['planet_type'].value_counts().index, palette='viridis')
    plt.title('Distribution of Planet Types')
    plt.savefig(config.VISUALIZATIONS_DIR / 'planet_type_distribution.png')
    plt.close()
    plt.figure(figsize=(12, 6))
    top_methods = df['discoverymethod'].value_counts().head(5).index
    sns.countplot(y='discoverymethod', data=df[df['discoverymethod'].isin(top_methods)], palette='magma')
    plt.title('Top 5 Discovery Methods')
    plt.savefig(config.VISUALIZATIONS_DIR / 'discovery_methods.png')
    plt.close()
    numerical_cols = config.FEATURES + [config.RADIUS_COL]
    plt.figure(figsize=(10, 8))
    corr = df[numerical_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Heatmap')
    plt.savefig(config.VISUALIZATIONS_DIR / 'correlation_heatmap.png')
    plt.close()
    # 4. Mass vs Orbital Period Scatter (Clearer alternative to pairplot)
    plt.figure(figsize=(12, 8))
    # Filter for positive values for log scale
    scatter_df = df[(df['pl_bmasse'] > 0) & (df['pl_orbper'] > 0)].copy()
    
    sns.scatterplot(
        data=scatter_df, 
        x='pl_orbper', 
        y='pl_bmasse', 
        hue='planet_type', 
        palette='viridis',
        alpha=0.7,
        edgecolor='w',
        s=80
    )
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Orbital Period [days] (Log Scale)')
    plt.ylabel('Planet Mass [Earth Mass] (Log Scale)')
    plt.title('Exoplanet Population: Mass vs. Period')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(config.VISUALIZATIONS_DIR / 'mass_vs_period_scatter.png')
    plt.close()
    print(f"EDA plots saved to {config.VISUALIZATIONS_DIR}")
run_eda()
