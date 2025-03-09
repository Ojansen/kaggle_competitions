from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.ollama import Ollama
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_json('data/all_plant_details.json')

# Define estimated room parameters (simulated labels for now)
# Assume a room with medium light, moderate humidity, and normal temperature
# Create a simple target variable: 1 (fit) if indoor & tolerates part shade, 0 otherwise

# df["fit_for_room"] = ((df["indoor"] == 1) & (df["sunlight_part_shade"] == 1)).astype(int)
#
# # Select features relevant to indoor suitability & room conditions
# features = [
#     "drought_tolerant", "salt_tolerant", "tropical", "indoor",
#     "sunlight_part_sun_part_shade", "sunlight_full_shade", "sunlight_deep_shade",
#     "sunlight_part_shade", "sunlight_full_sun_only_if_soil_kept_moist",
#     "watering_encoded", "care_level_encoded", "maintenance_encoded"
# ]
#
# X = df[features]
# y = df["fit_for_room"]
#
# # Split into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Train a basic Random Forest model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
#
# # Evaluate model performance
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)

llm = Ollama(model='granite3.2:8b')
print('Plant prediction model')
temp = input('Room temperature: ')
humidity = input('Room humidity: ')
light = input('Room light (hours of day): ')
radiator = input('Room radiator yes/no: ')
window = input('Room window yes/no: ')


prompt = f"""
You are a expert botanist that recommends plants for rooms based on room parameters. 
Return a list of plants that are suitable for the room. with a short description of each plant.

Room parameters:
- watering frequency: once a week
- temperature: {temp} degrees Celsius
- humidity: {humidity}%
- sun hours: {light}
- direct sunlight: yes
- radiator: {radiator}
- window: {window}
- on windowstill: yes

Plant database:
{df}
"""

result = llm.chat([
    ChatMessage(role="system", content=prompt),
    ChatMessage(role="user", content="What is a good plant for this room?")
])

print(result)

# print(accuracy)