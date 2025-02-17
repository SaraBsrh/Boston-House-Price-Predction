import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("/Applications/ML/Codes/Housing.csv")
print('Initial datset preview:')

# Define categorical mappings in a centralized dictionary
CATEGORICAL_MAPPING = {
    # Binary features
    'binary_columns': [
        'mainroad','guestroom', 'basement', 
        'hotwaterheating', 'airconditioning',
       'parking', 'prefarea'
    ],
    'binary_mapping': {'yes':1, 'no':0},

    # Ordinal feature (inherent order)
    'furnishingstatus': {
        'furnished':2,
        'semi-furnished':1, 
        'unfurnished':0
        }
}

# Convert binary features using pandas factorize (auto_detect categories)
for col in CATEGORICAL_MAPPING['binary_columns']:
    df[col] = df[col].replace(CATEGORICAL_MAPPING['binary_mapping'])

# Convert furnishing status using explicit mapping
df['furnishingstatus'] = df['furnishingstatus'].map(CATEGORICAL_MAPPING['furnishingstatus']).astype('int8')

# Standardize features
# columns = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
#        'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
#        'parking', 'prefarea', 'furnishingstatus']
# Scaler = StandardScaler()
# df[columns] = Scaler.fit_transform(df[columns])

features = ['area', 'bedrooms','bathrooms', 'stories','hotwaterheating', 'airconditioning',
       'parking', 'furnishingstatus']
X = df[features].values
y = df['price'].values

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1 , random_state=2)

# Create polynomial regression pipeline
degree = 2
poly_model = make_pipeline(
    PolynomialFeatures(degree),
    StandardScaler(),
    LinearRegression()
)

poly_model.fit(X_train, y_train)
y_test_pred = poly_model.predict(X_test[:, :8])
y_test = y_test

rss = np.mean(y_test - y_test_pred) ** 2
r2 = r2_score(y_test, y_test_pred)

print("Polynomial Regression Evaluation:")
print("Residual sum of squares: %.2f" % rss)
print('Variance score: %.2f' % r2)

joblib.dump(poly_model, "Codes/house_price_model.pkl")
print("saved")