# Plant API JSON Documentation

## Plant Object Fields

### Basic Information
- `id`: Unique identifier for the plant
- `common_name`: Common/popular name of the plant
- `scientific_name`: Array of scientific names (Latin names)
- `other_name`: Array of alternative common names
- `family`: Plant family name (e.g., "Pinaceae", "Sapindaceae")
- `genus`: Genus name (e.g., "Abies", "Acer")

### Taxonomic Classification
- `species_epithet`: The species part of the scientific name
- `cultivar`: Cultivated variety name
- `variety`: Botanical variety name
- `subspecies`: Subspecies name if applicable
- `hybrid`: Hybrid information if applicable
- `authority`: Botanical authority information

### Image Data
- `default_image`: Object containing image information:
  - `license`: License identifier
  - `license_name`: Name of the license
  - `license_url`: URL to license details
  - `original_url`: URL to original image
  - `regular_url`: URL to regular-sized image
  - `medium_url`: URL to medium-sized image
  - `small_url`: URL to small-sized image
  - `thumbnail`: URL to thumbnail image

### Origin and Type
- `origin`: Array of geographical origins
- `type`: Plant type (e.g., "tree")
- `cycle`: Life cycle type (e.g., "Perennial")

### Physical Characteristics
- `dimension`: Text description of plant size
- `dimensions`: Structured size data:
  - `type`: Measurement type
  - `min_value`: Minimum size
  - `max_value`: Maximum size
  - `unit`: Unit of measurement

### Growing Requirements
- `hardiness`: Plant hardiness zones:
  - `min`: Minimum zone
  - `max`: Maximum zone
- `hardiness_location`: URLs for hardiness zone maps
- `watering`: Watering frequency description
- `sunlight`: Array of sunlight requirements
- `soil`: Array of soil requirements
- `growth_rate`: Growth rate description
- `care_level`: Overall care difficulty

### Maintenance
- `pruning_month`: Array of recommended pruning months
- `maintenance`: Maintenance requirements
- `care-guides`: URL to care guides

### Features and Characteristics
- `drought_tolerant`: Boolean indicating drought tolerance
- `salt_tolerant`: Boolean indicating salt tolerance
- `thorny`: Boolean indicating presence of thorns
- `invasive`: Boolean indicating if plant is invasive
- `tropical`: Boolean indicating if plant is tropical
- `indoor`: Boolean indicating suitability as indoor plant
- `flowers`: Boolean indicating if plant produces flowers
- `flowering_season`: Flowering season if applicable
- `flower_color`: Flower colors
- `leaf`: Boolean indicating presence of leaves
- `leaf_color`: Array of leaf colors

### Safety and Edibility
- `poisonous_to_humans`: Boolean indicating toxicity to humans
- `poisonous_to_pets`: Boolean indicating toxicity to pets
- `edible_fruit`: Boolean indicating if fruit is edible
- `edible_leaf`: Boolean indicating if leaves are edible
- `medicinal`: Boolean indicating medicinal properties

### Description
- `description`: Detailed text description of the plant

### Additional Media
- `other_images`: Information about additional images (may require subscription)