import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


class PlantRecommendationSystem:
    def __init__(self, dataset_path):
        # Load the dataset
        self.df = pd.read_csv(dataset_path)

        # Preprocessing
        self.preprocess_data()

        # Feature columns for recommendation
        self.feature_columns = [
            'indoor',
            'sunlight_full_sun',
            'sunlight_part_sun_part_shade',
            'sunlight_full_shade',
            'drought_tolerant',
            'care_level_encoded',
            'maintenance_encoded',
            'watering_encoded'
        ]

    def preprocess_data(self):
        # Handle missing values
        self.df.fillna(0, inplace=True)

        # Normalize categorical features
        categorical_cols = [
            'indoor',
            'sunlight_full_sun',
            'sunlight_part_sun_part_shade',
            'sunlight_full_shade',
            'drought_tolerant'
        ]

        # Ensure binary encoding
        for col in categorical_cols:
            self.df[col] = self.df[col].astype(int)

    def create_user_preference_vector(self, preferences):
        """
        Create a feature vector from user preferences

        Expected preferences dictionary:
        {
            'indoor': bool,
            'sunlight': 'full_sun'/'part_sun'/'full_shade',
            'drought_tolerance': bool,
            'care_level': int (0-3),
            'maintenance_level': int (0-3),
            'watering_frequency': int (0-3)
        }
        """
        # Default preference vector
        user_vector = {
            'indoor': 1,
            'sunlight_full_sun': 0,
            'sunlight_part_sun_part_shade': 0,
            'sunlight_full_shade': 0,
            'drought_tolerant': 0,
            'care_level_encoded': 1,
            'maintenance_encoded': 1,
            'watering_encoded': 1
        }

        # Update user vector based on preferences
        if preferences.get('indoor') is not None:
            user_vector['indoor'] = int(preferences['indoor'])

        # Sunlight mapping
        if preferences.get('sunlight') == 'full_sun':
            user_vector['sunlight_full_sun'] = 1
        elif preferences.get('sunlight') == 'part_sun':
            user_vector['sunlight_part_sun_part_shade'] = 1
        elif preferences.get('sunlight') == 'full_shade':
            user_vector['sunlight_full_shade'] = 1

        # Other preferences
        if preferences.get('drought_tolerance') is not None:
            user_vector['drought_tolerant'] = int(preferences['drought_tolerance'])

        # Encoding levels
        if preferences.get('care_level') is not None:
            user_vector['care_level_encoded'] = preferences['care_level']

        if preferences.get('maintenance_level') is not None:
            user_vector['maintenance_encoded'] = preferences['maintenance_level']

        if preferences.get('watering_frequency') is not None:
            user_vector['watering_encoded'] = preferences['watering_frequency']

        return user_vector

    def calculate_compatibility(self, user_vector):
        """
        Calculate compatibility scores for all plants
        """
        # Convert user vector to DataFrame
        user_df = pd.DataFrame([user_vector])

        # Prepare plant features
        plant_features = self.df[self.feature_columns]

        # Calculate cosine similarity
        similarities = cosine_similarity(user_df[self.feature_columns], plant_features)[0]

        # Create results DataFrame
        results = self.df.copy()
        results['compatibility_score'] = similarities

        # Sort by compatibility score
        recommended_plants = results.sort_values('compatibility_score', ascending=False)

        return recommended_plants[['common_name', 'compatibility_score']].head(10)

    def recommend_plants(self, preferences):
        """
        Main recommendation method
        """
        # Create user preference vector
        user_vector = self.create_user_preference_vector(preferences)

        # Calculate and return recommendations
        return self.calculate_compatibility(user_vector)


# Example usage
def main():
    # Initialize recommendation system
    recommender = PlantRecommendationSystem('data/all_plant_details.csv')

    # Example user preferences
    user_preferences = {
        'indoor': True,
        'sunlight': 'full_sun',
        'drought_tolerance': True,
        'care_level': 1,
        'maintenance_level': 1,
        'watering_frequency': 1
    }

    # Get recommendations
    recommendations = recommender.recommend_plants(user_preferences)

    print("Top Plant Recommendations:")
    print(recommendations)


if __name__ == "__main__":
    main()