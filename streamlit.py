import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder

# Load the machine learning model and encode
model = joblib.load('XG-class.pkl')
gender_encode= joblib.load('gender_encode.pkl')
cr_card_encode = joblib.load('cr_card_encode.pkl')
act_member_encode = joblib.load('act_member_encode.pkl')


def main():
    st.title('Churn Model Deployment')
    
    Surname = st.text_input("Surname: ")
    Geography = st.radio("Geography: ", ["France","Germany", 'Spain'])
    CreditScore = st.number_input("Credit Score :", 300,900)
    Age = st.number_input("Input Age", 0, 100)
    Gender = st.radio("Input Gender : ", ["Male","Female"])
    Tenure = st.number_input("the period of time you holds a position (in years)", 0,100)
    Balance = st.number_input("Balance :")
    NumOfProducts = st.number_input("Number Of Products :")
    HasCrCard = st.radio("I Have a Credit Card : ", ["Yes","No"])
    IsActiveMember = st.radio("I am an Active Member : ", ["Yes","No"])
    EstimatedSalary = st.number_input("Estimated Salary :")

    geography_encoding = {'France': [0, 0, 1], 'Spain': [0, 1, 0], 'Germany': [1, 0, 0]}
    geography_encoded = geography_encoding[Geography]

    
    data = {'Age': int(Age), 'Gender': Gender, 
            'CreditScore':int(CreditScore),
            'Tenure': int(Tenure), 'Balance':int(Balance),
            'NumOfProducts': NumOfProducts, 'HasCrCard': HasCrCard,
            'IsActiveMember':IsActiveMember,'EstimatedSalary':int(EstimatedSalary)}

    
    df=pd.DataFrame([list(data.values())], columns=['CreditScore','Gender',  
                                                'Age', 'Tenure','Balance', 
                                                'NumOfProducts', 'HasCrCard' ,'IsActiveMember', 'EstimatedSalary',])
    
    scaler = RobustScaler()

    df=df.replace(gender_encode)
    df=df.replace(cr_card_encode)
    df=df.replace(act_member_encode)   

    df = scaler.fit_transform(df)
    
    if st.button('Make Prediction'):
        features=df      
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()
