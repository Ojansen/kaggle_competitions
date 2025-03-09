import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path

def load_plant_data():
    """Load and clean the plant details data."""
    with open('data/all_plant_details.json', 'r') as f:
        data = json.load(f)
    
    # Filter out any non-dictionary entries or entries without ID
    data = [d for d in data if isinstance(d, dict) and 'id' in d]
    return data

def flatten_list_fields(data):
    """Convert list fields to comma-separated strings for DataFrame."""
    flattened = []
    for plant in data:
        plant_copy = plant.copy()
        for key, value in plant_copy.items():
            if isinstance(value, list):
                plant_copy[key] = ', '.join(str(v) for v in value) if value else None
        flattened.append(plant_copy)
    return flattened

def analyze_plants():
    """Perform EDA on plant details data."""
    # Create output directory
    Path('analysis/outputs').mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    data = load_plant_data()
    df = pd.DataFrame(flatten_list_fields(data))
    
    print(f"Total number of plants: {len(df)}")
    
    # Basic statistics and info
    print("\nBasic Dataset Information:")
    print(df.info())
    
    # Save basic statistics to file
    with open('analysis/outputs/basic_stats.txt', 'w') as f:
        f.write("Dataset Statistics\n")
        f.write("=================\n\n")
        f.write(f"Total number of plants: {len(df)}\n\n")
        f.write("Categorical Variables Summary:\n")
        
        # Analyze categorical variables
        categorical_cols = ['cycle', 'watering', 'maintenance', 'growth_rate', 'care_level']
        for col in categorical_cols:
            if col in df.columns:
                value_counts = df[col].value_counts()
                f.write(f"\n{col.title()} Distribution:\n")
                f.write(value_counts.to_string())
                f.write("\n")
                
                # Create pie charts
                plt.figure(figsize=(10, 6))
                plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
                plt.title(f'Distribution of {col.title()}')
                plt.savefig(f'analysis/outputs/{col}_distribution.png')
                plt.close()

    # Analyze boolean characteristics
    boolean_cols = ['indoor', 'tropical', 'drought_tolerant', 'poisonous_to_humans', 
                   'poisonous_to_pets', 'medicinal', 'edible_leaf', 'flowers']
    
    plt.figure(figsize=(15, 8))
    boolean_data = df[boolean_cols].sum() / len(df) * 100
    boolean_data.plot(kind='bar')
    plt.title('Percentage of Plants with Various Characteristics')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('analysis/outputs/plant_characteristics.png')
    plt.close()

    # Analyze sunlight preferences
    sunlight_data = []
    for sunlight in df['sunlight']:
        if pd.notna(sunlight):
            sunlight_data.extend([s.strip() for s in sunlight.split(',')])
    
    sunlight_counts = pd.Series(Counter(sunlight_data))
    plt.figure(figsize=(12, 6))
    sunlight_counts.plot(kind='bar')
    plt.title('Distribution of Sunlight Requirements')
    plt.xlabel('Sunlight Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('analysis/outputs/sunlight_distribution.png')
    plt.close()

    # Analyze leaf colors
    leaf_colors = []
    for colors in df['leaf_color']:
        if pd.notna(colors):
            leaf_colors.extend([c.strip() for c in colors.split(',')])
    
    leaf_color_counts = pd.Series(Counter(leaf_colors))
    plt.figure(figsize=(10, 6))
    leaf_color_counts.plot(kind='bar')
    plt.title('Distribution of Leaf Colors')
    plt.xlabel('Color')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('analysis/outputs/leaf_color_distribution.png')
    plt.close()

    # Correlation analysis for numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        correlation = df[numeric_cols].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Numeric Variables')
        plt.tight_layout()
        plt.savefig('analysis/outputs/correlation_matrix.png')
        plt.close()

    # Save summary to markdown
    with open('analysis/outputs/summary.md', 'w') as f:
        f.write("# Plant Database Analysis Summary\n\n")
        f.write(f"## Dataset Overview\n")
        f.write(f"- Total number of plants: {len(df)}\n")
        f.write(f"- Number of indoor plants: {df['indoor'].sum()}\n")
        f.write(f"- Number of toxic plants: {df['poisonous_to_humans'].sum()}\n\n")
        
        f.write("## Key Findings\n")
        f.write("1. Most Common Characteristics:\n")
        for col in boolean_cols:
            percentage = (df[col].sum() / len(df) * 100)
            f.write(f"   - {col.replace('_', ' ').title()}: {percentage:.1f}%\n")
        
        f.write("\n2. Care Level Distribution:\n")
        if 'care_level' in df.columns:
            care_dist = df['care_level'].value_counts()
            for level, count in care_dist.items():
                f.write(f"   - {level}: {count} plants ({count/len(df)*100:.1f}%)\n")

if __name__ == "__main__":
    analyze_plants()