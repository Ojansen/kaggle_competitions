import json
import time
from pathlib import Path
import requests


def fetch_all_plants():
    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)

    all_plants = []
    current_page = 1
    last_page = 13  # From your selected code

    print(f"Fetching {last_page} pages of plants...")

    for page in range(1, last_page + 1):
        url = f"https://perenual.com/api/v2/species-list?page={page}&key=sk-UYl467c62c8acabe38942&indoor=1"
        response = requests.get(url)
        data = response.json()

        if 'data' in data:
            all_plants.extend(data['data'])
            print(f"Fetched page {page}/{last_page} - Got {len(data['data'])} plants")
        else:
            print(f"Error fetching page {page}: {data}")

        # Be nice to the API - add a small delay between requests
        time.sleep(0.5)

    # Save all plants list
    with open('data/all_plants.json', 'w') as f:
        json.dump(all_plants, f, indent=2)

    print(f"\nSaved {len(all_plants)} plants to data/all_plants.json")
    return all_plants


def fetch_all_details(resume_from_id=None):
    # Load plants from JSON file
    try:
        with open('data/all_plants.json', 'r') as f:
            plants = json.load(f)
    except FileNotFoundError:
        print("Error: data/all_plants.json not found. Please run fetch_all_plants() first.")
        return []

    # Load existing details if any
    details = []
    try:
        with open('data/all_plant_details.json', 'r') as f:
            details = json.load(f)
            # Filter out rate limit error responses
            details = [d for d in details if isinstance(d, dict) and 'id' in d]
            print(f"Loaded {len(details)} existing plant details")
    except FileNotFoundError:
        print("No existing details file found. Starting fresh.")

    # Get set of already fetched plant IDs
    fetched_ids = {d['id'] for d in details if isinstance(d, dict) and 'id' in d}

    # Filter plants to fetch
    if resume_from_id:
        plants = [p for p in plants if p['id'] >= resume_from_id]
    
    # Further filter out already fetched plants
    plants_to_fetch = [p for p in plants if p['id'] not in fetched_ids]
    
    total = len(plants_to_fetch)
    print(f"\nFetching details for {total} remaining plants...")

    for i, plant in enumerate(plants_to_fetch, 1):
        plant_id = plant['id']
        url = f"https://perenual.com/api/species/details/{plant_id}?key=sk-SEoj67c6cb8a7a2e38946"

        try:
            response = requests.get(url)
            data = response.json()

            print(response)
            
            # Check if we hit the rate limit
            if isinstance(data, dict) and 'X-RateLimit-Exceeded' in data:
                print(f"\nRate limit exceeded. Last successful ID: {plant_id - 1}")
                print(f"Resume later using: fetch_all_details(resume_from_id={plant_id})")
                break
                
            details.append(data)
            print(f"Fetched details for plant {i}/{total} (ID: {plant_id})")
        except Exception as e:
            print(f"Error fetching details for plant ID {plant_id}: {e}")
            continue

        # Save progress after each successful fetch
        with open('data/all_plant_details.json', 'w') as f:
            json.dump(details, f, indent=2)

        # Be nice to the API - add a small delay between requests
        time.sleep(0.5)

    print(f"\nSaved details for {len(details)} plants to data/all_plant_details.json")
    return details
