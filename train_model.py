import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Load your updated CSV
df = pd.read_csv("train_sample_with_location_type.csv")

# Ensure consistent casing in column names
df.columns = df.columns.str.lower()

# Feature and target
X = df[["location", "housetype", "bedroomabvgr", "fullbath", "lotarea", "garagearea", "1stflrsf", "yearbuilt"]]
y = df["saleprice"]

# Categorical & numerical columns
categorical = ["location", "housetype"]
numerical = ["bedroomabvgr", "fullbath", "lotarea", "garagearea", "1stflrsf", "yearbuilt"]

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical),
    ("num", StandardScaler(), numerical)
])

# Full pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
pipeline.fit(X_train, y_train)

# Save the model using pickle
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved to model.pkl")
