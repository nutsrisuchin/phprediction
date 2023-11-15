import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from joblib import load


# Load your trained model (for the sake of this example, we're initializing a dummy model)
# Make sure to replace this with loading your actual trained model
#model = LogisticRegression()

#model = load("C:\\Users\\26005064\\OneDrive - PTT Global Chemical Public Company Limited\\DATA\\Nut\\FiT, T-II, Digital Projects\\Sour water pH analysis\\production_model.pkl")
model = load("production_model.pkl")

st.title("pH Prediction App")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

if uploaded_file:
    data = pd.read_excel(uploaded_file, engine="openpyxl")
    st.write("Uploaded Data:")
    st.write(data)

    data_test_final = data.iloc[:, 2:]

    #data1 = data.iloc[:, :5]
    #data_concat = pd.concat([data1, pd.DataFrame(predicted_ph)], axis=1)
    #data_concat.rename(columns={data_concat.columns[-1]: 'IOW (In=1/Out=0)'}, inplace=True)

    # Predict pH values
    # For the sake of this example, we'll assume 'data' can be directly fed to the model.
    # Adjust preprocessing if necessary.
    predictions = model.predict(data_test_final)
    
    predictions_df = pd.DataFrame(predictions, columns=['IOW (In=1/Out=0)'])
    
    #st.write("Predictions:")
    #st.write(predictions)

    data1 = data.iloc[:, :2]
    data_concat = pd.concat([data1, predictions_df], axis=1)
    #data_concat = data_concat.rename(columns={data_concat.columns[-1]: 'IOW (In=1/Out=0)'}, inplace=True)
    st.write("Predictions:")
    st.write(data_concat)