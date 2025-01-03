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
   git clone https://github.com/Gulverda/Scraper.git
2. Navigate to the project directory:
  ```bash
  cd Scraper
  ```
3. Install dependencies:
  ```
  pip install -r requirements.txt
  ```

## Usage
1. Web Scraping
   ```
   python main.py
   ```
2.  Data Analysis and Visualization
    Preprocess data using the cleaning functions.
     - Use provided scripts to generate visualizations, such as:
     - Price distribution histograms.
     - Fuel type bar charts.
     - Transmission type pie charts.

3. Machine Learning
   Train and evaluate machine learning models:

      - Classification: Use Random Forest Classifier to categorize car prices into "Low," "Medium," or "High."
      - Regression: Use Linear Regression to predict car prices. Evaluate using metrics such as:
        - Mean Squared Error (MSE)
        - R-squared (R²)
     
## API and Libraries
   The project relies on the following libraries:

  - requests - For HTTP requests to the website.
  - beautifulsoup4 - For parsing HTML content.
  - json - For storing data in JSON format.
  - pandas - For data manipulation and preprocessing.
  - matplotlib - For basic visualizations.
  - seaborn - For advanced visualizations.
  - scikit-learn - For machine learning tasks (classification and regression).


##Outputs
  - Scraped Data: Saved in car_listings.json.
  - Visualizations: Generated as images displayed in the terminal or saved as files.
  - Model Metrics:
    - Classification: Confusion matrix, classification report.
    - Regression: MSE, R² score, and scatter plots.
      
## License

This project is licensed under the [MIT License](LICENSE).
