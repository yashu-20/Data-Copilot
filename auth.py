import streamlit as st

def login():
    password = st.text_input("Enter password to access:", type="password")
    return password == "admin"
