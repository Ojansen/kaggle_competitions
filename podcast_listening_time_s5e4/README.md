# Podcast Listening Time Prediction - Kaggle Competition

## Overview

This project is part of a Kaggle competition where the goal is to predict the listening time (in minutes) for podcast episodes based on various features such as episode length, genre, host popularity, and more. The dataset includes both training and test data, and the predictions are submitted to Kaggle for evaluation.

### Dataset

The dataset consists of the following files:

* train.csv: Contains the training data with features and the target variable (Listening_Time_minutes).
* test.csv: Contains the test data with features but without the target variable. Predictions are made on this dataset.
sample_submission.csv: A sample submission file showing the required format for predictions.

### Data Fields

* id: Unique identifier for each podcast episode.
* Podcast_Name: Name of the podcast.
* Episode_Title: Title of the episode.
* Episode_Length_minutes: Length of the episode in minutes.
* Genre: Genre of the podcast (e.g., Comedy, True Crime, Education).
* Host_Popularity_percentage: Popularity of the host as a percentage.
* Publication_Day: Day of the week the episode was published.
* Publication_Time: Time of day the episode was published (e.g., Morning, Afternoon).
* Guest_Popularity_percentage: Popularity of the guest as a percentage.
* Number_of_Ads: Number of advertisements in the episode.
Episode_Sentiment: Sentiment of the episode (e.g., Positive, Neutral, Negative).
* Listening_Time_minutes (only in train.csv): Target variable representing the listening time in minutes.

### Getting the Data

The dataset can be downloaded from the Kaggle competition page. Follow these steps to download the data:

1. Go to the competition page: Kaggle Competition - Podcast Listening Time.

2. Sign in to your Kaggle account.
3. Accept the competition rules.
4. Download the dataset and place the files in the data/ directory of this project.

### How to Run

1. Install Dependencies: Ensure you have Python 3.8+ and the required libraries installed. You can install the dependencies using:

    _(Note: Create a requirements.txt file if not already present.)_

```bash
pip install -r requirements.txt
```

2. Run the Script: Execute the main script to preprocess the data, train the model, and generate the submission file:

```bash
python main.py
```

