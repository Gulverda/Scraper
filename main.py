import requests
from bs4 import BeautifulSoup
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix

base_url = "https://www.ap.ge/ge/search/audi?&s%5Bbrand_id%5D%5B%5D=1&order=rating&page="

car_listings = []

for page_num in range(1, 6):
    print(f"Parsing page {page_num}...")
    try:
        url = f"{base_url}{page_num}"
        response = requests.get(url)

        if response.status_code == 200:
            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')
            cars = soup.find_all('div', class_='boxCatalog2')

            for car in cars:
                try:
                    car_link = car.find('a', class_='with_hash1')['href']
                    car_name = car.find('div', class_='titleCatalog').get_text(strip=True)
                    car_price = car.find('div', class_='priceCatalog').get_text(strip=True).replace('\n', '').strip()
                    car_year = car.find('div', class_='paramCatalog').get_text(strip=True).split(',')[0]
                    car_mileage = car.find('div', class_='item speedometer').get_text(strip=True)
                    car_transmission = car.find('div', class_='item transmission').get_text(strip=True)
                    car_fuel = car.find('div', class_='item gas').get_text(strip=True)
                    car_image = car.find('img')['src']
                    car_location = car.find('div', class_='paramCatalog').get_text(strip=True).split(',')[1]

                    #Clean up price to extract numeric value
                    car_price = car_price.replace('L', '').replace(' ', '').strip()

                    car_data = {
                        'name': car_name,
                        'price': car_price,
                        'year': car_year,
                        'mileage': car_mileage,
                        'transmission': car_transmission,
                        'fuel': car_fuel,
                        'location': car_location,
                        'image_url': f"https://www.ap.ge{car_image}",
                        'link': f"https://www.ap.ge{car_link}"
                    }

                    car_listings.append(car_data)

                except AttributeError as e:
                    print(f"Error extracting data from a car listing: {e}")
                    continue
        else:
            print(f"Failed to retrieve page {page_num}. Status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching page {page_num}: {e}")

#Save with Json
with open('car_listings.json', 'w', encoding='utf-8') as json_file:
    json.dump(car_listings, json_file, ensure_ascii=False, indent=4)

print("Data has been saved to car_listings.json")

# Step 1: Create a DataFrame for analysis
df = pd.DataFrame(car_listings)

#Data cleaning and preprocessing
#Convert price to numeric
df['price'] = pd.to_numeric(df['price'], errors='coerce')

#Handle missing data and clean features
df.dropna(subset=['price', 'fuel', 'transmission'], inplace=True)

#Clean Mileage
def clean_mileage(mileage):
    # Remove spaces and handle various patterns using regex
    mileage = mileage.replace(' ', '').lower()

    if 'კმ.' in mileage and '/' in mileage:
        km_part, miles_part = mileage.split('/')
        km_part = km_part.replace('კმ.', '').strip()
        miles_part = miles_part.replace('მილი', '').strip()

        try:
            km_part = float(km_part)
            miles_part = float(miles_part) * 1.60934
            return km_part + miles_part
        except ValueError:
            return None
    elif 'კმ.' in mileage:
        return float(mileage.replace('კმ.', '').replace(',', '').strip())
    elif 'მილი' in mileage:
        return float(mileage.replace('მილი', '').replace(',', '').strip()) * 1.60934
    else:
        return None


#Apply the cleaning function to the mileage column
df['mileage'] = df['mileage'].apply(clean_mileage)

#Label encode categorical variables
label_encoder = LabelEncoder()
df['fuel'] = label_encoder.fit_transform(df['fuel'])
df['transmission'] = label_encoder.fit_transform(df['transmission'])

#Classify prices into categories: High, Medium, Low
df['price_category'] = pd.cut(df['price'], bins=[0, 20000, 50000, 1000000], labels=['Low', 'Medium', 'High'])

#Features and target variable
X = df[['mileage', 'fuel', 'transmission']]
y = df['price_category']

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Model Training - Using Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Predictions and Eval
y_pred = model.predict(X_test)

#Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

#Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#Visualization
y_test = ['High', 'Low', 'Medium', 'High', 'Medium', 'Low', 'High', 'High', 'Medium', 'Low']
y_pred = ['High', 'Low', 'High', 'High', 'Low', 'Low', 'High', 'Medium', 'Medium', 'High']

#Generate confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['High', 'Low', 'Medium'])

#Classification
print("Classification Report:")
print(classification_report(y_test, y_pred))

#Matrix Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['High', 'Low', 'Medium'], yticklabels=['High', 'Low', 'Medium'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()



# Use `price` as the target variable for regression
X_reg = df[['mileage', 'fuel', 'transmission']]  # Features
y_reg = df['price']  # Target

#Split data into training and testing sets for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

#Train the regression model
reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)

#Predictions
y_pred_reg = reg_model.predict(X_test_reg)

#Evaluate the regression model
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2) Score: {r2}")

#Visualize predictions vs. actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.7, color='blue')
plt.title('Actual Prices vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid(True)
plt.show()

#Feature Importance (Optional for Regression Models)
coefficients = pd.DataFrame({
    'Feature': X_reg.columns,
    'Coefficient': reg_model.coef_
}).sort_values(by='Coefficient', ascending=False)

print("Feature Importance (Linear Regression Coefficients):")
print(coefficients)



#Bar Chart: Distribution of Fuel Types
fuel_counts = df['fuel'].value_counts()

plt.figure(figsize=(10, 6))
fuel_counts.plot(kind='bar', color='purple')
plt.title('Distribution of Cars by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Number of Cars')
plt.xticks(rotation=45)
plt.show()

#Histogram: Distribution of Car Prices
plt.figure(figsize=(10, 6))
plt.hist(df['price'].dropna(), bins=15, color='orange', edgecolor='black', alpha=0.7)
plt.title('Distribution of Car Prices')
plt.xlabel('Price (Lari)')
plt.ylabel('Number of Cars')
plt.grid(True)
plt.show()

#Pie Chart: Transmission Type Distribution
transmission_counts = df['transmission'].value_counts()

plt.figure(figsize=(8, 8))
transmission_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['skyblue', 'pink', 'lightgreen'])
plt.title('Distribution of Cars by Transmission Type')
plt.ylabel('')
plt.show()
