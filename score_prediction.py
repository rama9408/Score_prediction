import streamlit as st
import numpy as np
import pandas as pd
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- 1. Load and Prepare Data ---
# Reading the dataset using pandas
df = pd.read_csv('score.csv')

# Making hours as feature(X) and Scores as Target(Y)
X = df.drop('Scores', axis=1)
y = df.Scores

# Splitting the dataset into train and test (80% train, 20% test)[cite: 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the linear regression model[cite: 1]
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

# Making predictions on the test set for evaluation[cite: 1]
y_pred = reg.predict(X_test)

# --- 2. Streamlit UI Elements ---
st.set_page_config(page_title="Student Score Predictor", layout="wide")
st.title("🎓 Student Score Prediction Dashboard")
st.markdown("Predict a student's score based on the number of hours they studied using Linear Regression.")

# --- 3. Sidebar for User Input ---
st.sidebar.header("Make a Prediction")
user_hours = st.sidebar.number_input("Enter Study Hours:", min_value=0.0, max_value=24.0, value=5.0, step=0.5)

# Predict based on user input
user_prediction = reg.predict([[user_hours]])[0]

# --- 4. Main Dashboard Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Model Performance Metrics")
    # Calculate evaluation metrics[cite: 1]
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Displaying metrics in neat tiles
    st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
    st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
    st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")

with col2:
    st.subheader("Prediction Result")
    st.write(f"For **{user_hours} hours** of study, the predicted score is:")
    st.title(f"{user_prediction:.2f}%")

# --- 5. Visualizations ---
st.divider()
st.subheader("Data Visualization & Regression Line")

fig, ax = plt.subplots(figsize=(10, 6))
# Scatter plot for actual data[cite: 1]
ax.scatter(df['Hours'], df['Scores'], color='blue', label='Actual Data Points')

# Plotting the Regression Line[cite: 1]
ax.plot(df['Hours'], reg.predict(X), color='red', linewidth=2, label='Regression Line')

# Highlighting the User's Input
ax.scatter(user_hours, user_prediction, color='green', s=150, edgecolors='black', zorder=5, label='Your Prediction')

ax.set_title('Hours Studied vs. Scores Obtained')
ax.set_xlabel('Hours Studied')
ax.set_ylabel('Scores Obtained')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

st.pyplot(fig)

# Show the raw data if the user wants to see it
if st.checkbox("Show Raw Data Table"):
    st.write(df)
