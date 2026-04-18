"""
Debug script to test if CSS injection works in Streamlit
"""

import streamlit as st

st.set_page_config(page_title="CSS Debug", layout="wide")

st.write("## Test 1: Plain st.markdown with CSS")

st.markdown("""
<style>
body { background: red !important; }
button { background: blue !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

st.write("If body is red and button is blue, CSS works")
st.button("Test Button", key="test1")

st.write("---")

st.write("## Test 2: With fonts")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;700&display=swap" rel="stylesheet">
<style>
* { font-family: 'Space Grotesk', sans-serif !important; }
</style>
""", unsafe_allow_html=True)

st.write("If text is Space Grotesk, fonts work")
st.button("Test Font Button", key="test2")
