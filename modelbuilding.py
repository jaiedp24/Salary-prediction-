import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv('Salary_Data.csv')

# Handle missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].mean())

# Encode categorical columns
encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# Split data
X = df.drop('Salary', axis=1)
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save encoders
with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

# Save column order
with open('columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ Model, encoders, and columns saved successfully!")

import streamlit as st
import pandas as pd
import pickle

st.title("💼 Salary Prediction App")

# Load files
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)

    with open('columns.pkl', 'rb') as f:
        columns = pickle.load(f)

    st.success("Model loaded successfully!")

except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# --- User Inputs ---
age = st.slider("Age", 18, 65, 30)

gender_input = st.selectbox("Gender", encoders['Gender'].classes_)
education_input = st.selectbox("Education Level", encoders['Education Level'].classes_)
job_input = st.selectbox("Job Title", encoders['Job Title'].classes_)

experience = st.slider("Years of Experience", 0.0, 40.0, 5.0)

# --- Encoding ---
try:
    gender = encoders['Gender'].transform([gender_input])[0]
    education = encoders['Education Level'].transform([education_input])[0]
    job = encoders['Job Title'].transform([job_input])[0]
except Exception as e:
    st.error(f"Encoding error: {e}")
    st.stop()

# --- Prediction ---
if st.button("Predict Salary"):
    try:
        input_data = pd.DataFrame([[age, gender, education, job, experience]], columns=columns)

        prediction = model.predict(input_data)[0]

        st.success(f"💰 Predicted Salary: ${prediction:,.2f}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
