# Car Listings Scraper and Analysis

## Overview

This project is designed for scraping car listings from [AP.ge](https://www.ap.ge), cleaning and preprocessing the data, and performing machine learning tasks such as classification and regression. It also includes visualizations for exploring the data and understanding trends.

---

## Features

### Web Scraping
- Scrapes car listings from [AP.ge](https://www.ap.ge/ge/search/audi), including:
  - Car name
  - Price
  - Year
  - Mileage
  - Transmission type
  - Fuel type
  - Location
  - Image URL
- Saves data to a JSON file (`car_listings.json`).

### Data Cleaning and Preprocessing
- Converts price to numeric format.
- Standardizes mileage values.
- Encodes categorical variables (e.g., fuel type, transmission type).
- Handles missing data by removing rows with essential missing values.

### Machine Learning Models
- **Random Forest Classifier**: Classifies cars into price ranges (Low, Medium, High).
- **Linear Regression**: Predicts car prices based on features like mileage, fuel type, and transmission type.

### Data Visualization
- Generates insightful visualizations:
  - Confusion matrix heatmaps for classification performance.
  - Scatter plots for regression analysis (actual vs. predicted prices).
  - Bar charts, histograms, and pie charts for exploratory data analysis.

---

## Installation

Follow the steps below to set up the project environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/car-listings-scraper.git
