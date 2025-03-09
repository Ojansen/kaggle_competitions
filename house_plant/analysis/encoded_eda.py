import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_encoded_data():
    # Create output directory
    Path('analysis/outputs/encoded').mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv('data/all_plant_details.csv')
    
    # Group columns by type
    boolean_cols = [col for col in df.columns if set(df[col].unique()).issubset({0, 1})]
    encoded_cols = ['care_level_encoded', 'maintenance_encoded', 'watering_encoded', 'growth_rate_encoded']
    sunlight_cols = [col for col in df.columns if col.startswith('sunlight_')]
    cycle_cols = [col for col in df.columns if col.startswith('cycle_')]

    # 1. Analyze encoded ordinal features
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(encoded_cols, 1):
        plt.subplot(2, 2, i)
        sns.histplot(data=df, x=col, bins=len(df[col].unique()))
        plt.title(f'Distribution of {col.replace("_encoded", "").replace("_", " ").title()}')
    plt.tight_layout()
    plt.savefig('analysis/outputs/encoded/ordinal_features_distribution.png')
    plt.close()

    # 2. Correlation matrix for encoded features
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[encoded_cols].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Encoded Features')
    plt.tight_layout()
    plt.savefig('analysis/outputs/encoded/encoded_correlation.png')
    plt.close()

    # 3. Boolean features analysis
    boolean_percentages = df[boolean_cols].mean().sort_values(ascending=False) * 100
    
    plt.figure(figsize=(15, 8))
    boolean_percentages.plot(kind='bar')
    plt.title('Percentage of Plants with Various Characteristics')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('analysis/outputs/encoded/boolean_features.png')
    plt.close()

    # 4. Sunlight preferences analysis
    sunlight_percentages = df[sunlight_cols].mean().sort_values(ascending=False) * 100
    
    plt.figure(figsize=(12, 6))
    sunlight_percentages.plot(kind='bar')
    plt.title('Distribution of Sunlight Requirements')
    plt.ylabel('Percentage of Plants')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('analysis/outputs/encoded/sunlight_distribution.png')
    plt.close()

    # 5. Cycle type analysis
    cycle_percentages = df[cycle_cols].mean().sort_values(ascending=False) * 100
    
    plt.figure(figsize=(10, 6))
    cycle_percentages.plot(kind='bar')
    plt.title('Distribution of Plant Life Cycles')
    plt.ylabel('Percentage of Plants')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('analysis/outputs/encoded/cycle_distribution.png')
    plt.close()

    # 6. Save summary statistics to markdown
    with open('analysis/outputs/encoded/summary.md', 'w') as f:
        f.write("# Encoded Plant Database Analysis\n\n")
        
        f.write("## Ordinal Features Statistics\n")
        for col in encoded_cols:
            f.write(f"\n### {col.replace('_encoded', '').replace('_', ' ').title()}\n")
            stats = df[col].describe()
            f.write(f"- Mean: {stats['mean']:.2f}\n")
            f.write(f"- Median: {stats['50%']:.2f}\n")
            f.write(f"- Std Dev: {stats['std']:.2f}\n")
            
        f.write("\n## Boolean Features (Percentage of True)\n")
        for feature, percentage in boolean_percentages.items():
            f.write(f"- {feature.replace('_', ' ').title()}: {percentage:.1f}%\n")
            
        f.write("\n## Sunlight Requirements\n")
        for condition, percentage in sunlight_percentages.items():
            f.write(f"- {condition.replace('sunlight_', '').replace('_', ' ').title()}: {percentage:.1f}%\n")
            
        f.write("\n## Life Cycles\n")
        for cycle, percentage in cycle_percentages.items():
            f.write(f"- {cycle.replace('cycle_', '').replace('_', ' ').title()}: {percentage:.1f}%\n")

        # Add correlation insights
        f.write("\n## Key Correlations\n")
        corr_matrix = df[encoded_cols].corr()
        for i in range(len(encoded_cols)):
            for j in range(i+1, len(encoded_cols)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.3:  # Only show meaningful correlations
                    f.write(f"- {encoded_cols[i].replace('_encoded', '')} and {encoded_cols[j].replace('_encoded', '')}: {corr:.2f}\n")

if __name__ == "__main__":
    analyze_encoded_data()