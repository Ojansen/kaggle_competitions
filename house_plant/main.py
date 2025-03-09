from api import fetch_all_plants, fetch_all_details
import pandas as pd
import ast
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv('data/all_plant_details.csv')

def ordinal_encode_care_features():
    # Define ordinal mappings (from lowest to highest)
    care_level_order = ['Easy', 'Moderate', 'Medium', 'High']
    maintenance_order = ['Low', 'Moderate', 'High']
    watering_order = ['Minimum', 'Average', 'Frequent']
    growth_rate_order = ['Low', 'Moderate', 'High']

    # Create ordinal encoder
    ordinal_encoder = OrdinalEncoder()

    # Create mapping dictionaries for each feature
    care_level_map = {level: idx for idx, level in enumerate(care_level_order)}
    maintenance_map = {level: idx for idx, level in enumerate(maintenance_order)}
    watering_map = {level: idx for idx, level in enumerate(watering_order)}
    growth_rate_map = {level: idx for idx, level in enumerate(growth_rate_order)}

    # Apply mappings
    df['care_level_encoded'] = df['care_level'].map(care_level_map)
    df['maintenance_encoded'] = df['maintenance'].map(maintenance_map)
    df['watering_encoded'] = df['watering'].map(watering_map)
    df['growth_rate_encoded'] = df['growth_rate'].map(growth_rate_map)

    # Print value distributions
    print("Care Level Encoding:")
    print(df['care_level_encoded'].value_counts().sort_index())
    print("\nMaintenance Encoding:")
    print(df['maintenance_encoded'].value_counts().sort_index())
    print("\nWatering Encoding:")
    print(df['watering_encoded'].value_counts().sort_index())
    print("\nGrowth Rate Encoding:")
    print(df['growth_rate_encoded'].value_counts().sort_index())

    # Save the encoded data
    df.to_csv('data/all_plant_details.csv', index=False)

    # Update features list with new encoded columns
    encoded_features = ['care_level_encoded', 'maintenance_encoded', 
                       'watering_encoded', 'growth_rate_encoded']
    return encoded_features

def convert_booleans():
    # Get all boolean columns
    boolean_cols = [
        'drought_tolerant', 'salt_tolerant', 'thorny', 'invasive',
        'tropical', 'indoor', 'flowers', 'cones', 'fruits',
        'edible_fruit', 'leaf', 'edible_leaf', 'medicinal',
        'poisonous_to_humans', 'poisonous_to_pets', 'cuisine', 'seeds'
    ]
    
    # Convert each boolean column to 0/1
    for col in boolean_cols:
        df[col] = df[col].astype(int)
    
    # Print value counts for verification
    print("Boolean columns converted to 0/1:")
    for col in boolean_cols:
        print(f"\n{col}:")
        print(df[col].value_counts().sort_index())
        print(f"Percentage of 1s: {(df[col].sum() / len(df) * 100):.1f}%")
    
    # Save the updated dataframe
    df.to_csv('data/all_plant_details.csv', index=False)
    
    return boolean_cols

numeric_cols = df.select_dtypes(include=['bool']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

features = []
X = df[features]


def analyze_sunlight_conditions():
    # Read the CSV file
    df = pd.read_csv('data/all_plant_details.csv')

    # Convert string representation of list to actual list
    def parse_sunlight(sun_str):
        try:
            return ast.literal_eval(sun_str)
        except:
            return []

    # Apply the parsing function
    df['sunlight'] = df['sunlight'].apply(parse_sunlight)

    # Get all unique sunlight conditions
    unique_conditions = set()
    for conditions in df['sunlight']:
        unique_conditions.update(conditions)

    print("Unique sunlight conditions:", len(unique_conditions))
    for condition in sorted(unique_conditions):
        print(f"- {condition}")

    # Create one-hot encoded columns
    for condition in unique_conditions:
        col_name = f"sunlight_{condition.lower().replace(' ', '_').replace('/', '_')}"
        df[col_name] = df['sunlight'].apply(lambda x: 1 if condition in x else 0)

    # Print sample of the one-hot encoded data
    print("\nSample of one-hot encoded data:")
    sunlight_cols = [col for col in df.columns if col.startswith('sunlight_')]
    print(df[sunlight_cols].head())

    # Print frequency of each sunlight condition
    print("\nFrequency of each sunlight condition:")
    for col in sunlight_cols:
        condition_name = col.replace('sunlight_', '').replace('_', ' ').title()
        count = df[col].sum()
        percentage = (count / len(df)) * 100
        print(f"{condition_name}: {count} plants ({percentage:.1f}%)")

    # Print common combinations
    print("\nCommon sunlight requirement combinations:")
    sunlight_combinations = df['sunlight'].value_counts().head(10)
    for combo, count in sunlight_combinations.items():
        percentage = (count / len(df)) * 100
        print(f"{list(combo)}: {count} plants ({percentage:.1f}%)")

    df.merge(df[sunlight_cols], left_index=True, right_index=True)
    df =  df.drop(columns=['sunlight'])
    # df.merge(df[sunlight_cols], left_index=True, right_index=True)

    df.to_csv('data/all_plant_details.csv', index=False)

    print(df.columns)


def one_hot_encode_cycle():
    # Get unique values in cycle column
    unique_cycles = df['cycle'].unique()
    print("Unique cycle values:", sorted(unique_cycles))

    # Create one-hot encoded columns
    for cycle in unique_cycles:
        if pd.notna(cycle):  # Skip NaN values
            col_name = f"cycle_{cycle.lower().replace(' ', '_')}"
            df[col_name] = (df['cycle'] == cycle).astype(int)

    # Print value counts for verification
    cycle_cols = [col for col in df.columns if col.startswith('cycle_')]
    print("\nCycle columns converted to one-hot encoding:")
    for col in cycle_cols:
        cycle_name = col.replace('cycle_', '').replace('_', ' ').title()
        count = df[col].sum()
        percentage = (count / len(df)) * 100
        print(f"\n{cycle_name}:")
        print(f"Count: {count}")
        print(f"Percentage: {percentage:.1f}%")

    # Drop original cycle column
    df.drop('cycle', axis=1, inplace=True)

    # Save the updated dataframe
    df.to_csv('data/all_plant_details.csv', index=False)

    return cycle_cols

if __name__ == "__main__":
    # cycle_cols = one_hot_encode_cycle()

    # df.merge(df[cycle_cols], left_index=True, right_index=True)
    # df = df.drop('cycle', axis=1)
    df = df.drop(columns=['propagation', 'type', 'family'])
    df.to_csv('data/all_plant_details.csv', index=False)
    # boolean_features = convert_booleans()
    # df.merge(df[boolean_features], left_index=True, right_index=True)
    # df.to_csv('data/all_plant_details.csv', index=False)
    # print(df.columns)
    # print(boolean_features)
    # encoded_features = ordinal_encode_care_features()
    # features = boolean_features + encoded_features
    # print("\nAll features ready for analysis:", features)
    # fetch_all_details(resume_from_id=2976)
    # for col in df[categorical_cols].columns:
    #     print(df[categorical_cols][col].unique())  # to print categories name only
    #     print(df[categorical_cols][col].value_counts())  # to print count of every category
    # print(df[numeric_cols].describe())
    # analyze_sunlight_conditions()

