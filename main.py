import langchain_helper as lch

import streamlit as st

st.title("Pets name generator")
animal_type = st.sidebar.selectbox("What is your pet type?", ("dog", "cat", "cow", "Hamster"))

pet_color = st.sidebar.text_area(label="What is your pet color?", max_chars=15)

response = lch.generate_pet_name(animal_type, pet_color)
st.text(response['pet_name'])