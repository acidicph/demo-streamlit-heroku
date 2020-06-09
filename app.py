import streamlit as st

st.title("My App")

clicked = st.button("Click Me")
if clicked:
   st.balloons()
