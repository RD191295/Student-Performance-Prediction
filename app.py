import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder



def load_model():
    """Loads the saved model and its dependencies from a pickle file.

    Loads the following from the file:
    - model: a LinearRegression model
    - scaler: a StandardScaler used to scale the input data
    - le: a LabelEncoder used to encode categorical data

    Returns:
    - model, scaler, le as a tuple
    """
    with open('model.pkl','rb') as file:
        model,scaler,le = pickle.load(file)
    return model,scaler,le

def proecessing_input_data(data,scaler,le):
    """Processes the input data to prepare it for the model.

    Args:
    - data: a dictionary containing the input data
    - scaler: a StandardScaler used to scale the input data
    - le: a LabelEncoder used to encode categorical data

    Returns:
    - a numpy array containing the processed data
    """
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])[0]
    data = pd.DataFrame([data])
    data_transformed = scaler.transform(data)
    return data

def predict_data(data):
    """Predicts the output based on the input data.

    Args:
    - data: a numpy array containing the input data
    - model: a LinearRegression model

    Returns:
    - the predicted output
    """
    model,scaler,le = load_model()
    processed_data = proecessing_input_data(data,scaler,le)
    prediction = model.predict(processed_data)
    return prediction


def main():

    st.title("Student Performance prediction")
    st.subheader("student performance prediction")
    hours_studied = st.number_input("Hours Studied",key="hours_studied", min_value = 1, max_value = 10)
    pre_score = st.number_input("Previous Scores",key="previous_scores", min_value = 40, max_value = 100, value = 40)
    exe_activity = st.selectbox("Extracurricular Activities",["Yes","No"],key="extracurricular_activities")
    sleep_hours = st.number_input("Sleep Hours",key="sleep_hours", min_value = 4, max_value = 10,value = 7)
    no_que_solved = st.number_input("Number of question papers solved",key="No_Que_solved", min_value = 4, max_value = 10,value = 7)

    if st.button("predict Index"):
        user_data = {
            "Hours Studied":hours_studied,
            "Previous Scores":pre_score,
            "Extracurricular Activities":exe_activity,
            "Sleep Hours":sleep_hours,
            "Sample Question Papers Practiced":no_que_solved
        }
        prediction = predict_data(user_data)
        st.success(f"your prediciotn result is {prediction}")


if __name__ == "__main__":
    main()
    
