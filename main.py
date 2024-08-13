import langchain_helper as lch
import streamlit as st
import textwrap

st.title("Youtube Assistant")

with st.sidebar:
    youtube_url = st.sidebar.text_area(
        label="What is your Youtube video URL?",
        max_chars= 50,
    )
    with st.form(key='my_form'):
        query = st.text_area(
            label="Ask me about your YouTube video",
            max_chars= 50,
            key="query"
        )
        submit_btn = st.form_submit_button(label='Submit')


if query and youtube_url:
    db = lch.create_vector_db_from_youtube_url(youtube_url)
    response, doc = lch.get_response_from_query(db, query)
    st.subheader("Answer")
    st.text(textwrap.fill(response, width=80))