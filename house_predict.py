import joblib
import numpy as np

model = joblib.load("Codes/house_price_model.pkl")

def get_user_input():
    print("Please enter the details of the house:")

    area = float(input("Enter area (sqft): "))
    bedrooms = int(input("Enter number of bedrooms: "))
    bathrooms = int(input("Enter number of bathrooms: "))
    stories = int(input("Enter number of stories: "))
    hotwaterheating = int(input("Does it have hot water heating? (1 for Yes, 0 for No): "))
    airconditioning = int(input("Does it have air conditioning? (1 for Yes, 0 for No): "))
    parking = int(input("Does it have parking? (1 for Yes, 0 for No): "))
    furnishingstatus = int(input("how is the furnishing status? (2 for furnished, 1 for semi-furnished, 0 for unfurnished): "))
    
    # Convert to numpy array for prediction
    user_data = np.array([[area,bedrooms, bathrooms, stories, hotwaterheating, airconditioning, parking, furnishingstatus]])

    return user_data

# Get input from user
user_input = get_user_input()

# Predict price
predicted_price = model.predict(user_input)

# Show result
print(f"Estimated House Price: ${predicted_price[0]:,.2f}")