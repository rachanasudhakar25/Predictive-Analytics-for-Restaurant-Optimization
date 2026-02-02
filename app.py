import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

scalar = StandardScaler()

st.set_page_config(layout= "wide")

st.title(" RESTRAUNT RATING PREDICTION APP")

st.caption("this app helps you to predict a restraunts review class")

st.divider()

averagecost = st.number_input("please enter the estimated averae cost for two")

tablebooking  = st.selectbox("restraunt has table booking?", ["yes","no"])

onlinedelivery = st.selectbox("resraunt has online booking?", ["yes","no"])

pricerange = st.selectbox("what is the price range (1 cheapest, 4 most expensive)", [1,2,3,4])

predictbutton = st.button("predict the review")

st.divider()

model = joblib.load("mlmodel.pkl")

bookingstatus = 1 if tablebooking == "yes" else 0

deliverystatus = 1 if onlinedelivery == "yes" else 0

values = [[averagecost,bookingstatus,deliverystatus,pricerange]]

my_X_values = np.array(values)

X = scalar.transform(my_X_values)

if predictbutton:
    st.snow()

    prediction = model.predict(X)

    if prediction < 2.5:
        st.write("poor")
    elif prediction < 3.5:
        st.write("average")
    elif prediction < 4.0:
        st.write("good")
    elif prediction < 4.5:
        st.write("very good")
    else:
        st.write("excellent")
        

