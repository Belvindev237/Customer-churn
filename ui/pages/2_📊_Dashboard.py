import streamlit as st
import numpy as np

st.set_page_config(page_title="Churn prediction Dashboard",layout="wide")
st.title("📊 Churn Prediction Dashboard")

container=st.container(border=True, height="content")
with container:
  row1=st.columns(4)
  with row1[0]:st.metric("KPI1","50",delta_color="inverse" , border=True  , delta="1.5")
  with row1[1]:st.metric("KPI2","89" , border=True , delta="5")
  with row1[2]:st.metric("KPI3","15" , border=True , delta="8.9")
  with row1[3]:st.metric("KPI4","50%" , border=True , delta="1.2 %")


  row2=st.columns(2)

  with row2[0]:
    with st.container(border=True):  # Bordure individuelle
        st.bar_chart(np.random.randn(50, 3))

  with row2[1]:
    with st.container(border=True):  # Bordure individuelle
        st.bar_chart(np.random.randn(50, 3))
row3=st.columns(2)
