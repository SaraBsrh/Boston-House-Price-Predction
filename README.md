# Boston-House-Price-Predction

🏠 House Price Prediction (ML Project)

📌 Predicting Realistic and Reliable House Prices Using Machine Learning

📖 Overview

This project aims to predict house prices accurately based on key features such as area, bathrooms, air conditioning, stories, parking, furnishing status, hot water heating, and bedrooms. The goal is to help buyers and sellers set a realistic and reliable price while understanding which features impact the price the most.

📊 Dataset
	•	Source: Kaggle
	•	Size: 545 houses, 13 columns
	•	Features Used for Training (X): 8 features
	•	Target Variable (Y): Price

⚙️ Data Preprocessing
	•	Checked for null values (none found).
	•	Converted yes/no columns to binary (0/1).
	•	Transformed furnishing status into numerical values.
	•	Standardized the dataset.
	•	Checked for outliers and visualized data using:
	•	Histograms
	•	Scatter Plots
	•	Pair Plots
	•	Violin & Box Plots

📈 Models Used & Performance
	•	Linear Regression: R² Score = 69%
	•	Polynomial Regression (degree=2): R² Score = 74% ✅ (Best Model)
	•	Evaluation Metrics: MSE (Mean Squared Error) & R² Score

🚀 How to Run the Project

1️⃣ Install Dependencies

Run the following command to install all required libraries:

pip install -r requirements.txt

2️⃣ Train the Model

Execute the following script to train and save the model:

python train_model.py

3️⃣ Make Predictions

Run the following script to input house details and get the predicted price:

python predict.py

🔧 Future Improvements
	•	Implement hyperparameter tuning to improve accuracy.
	•	Try advanced regression models (e.g., Random Forest, XGBoost).
	•	Deploy the model using Flask/FastAPI for real-world use.

📫 Contact

💻 GitHub: SaraBsrh
📧 Email: SaraBasereh83@gmail.com
