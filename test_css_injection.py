"""Test CSS injection methods."""
import streamlit as st

st.set_page_config(layout="wide")

st.write("## Method 1: st.html() with style tag")
st.html("""
<style>
body { background: red !important; }
h1 { color: blue !important; }
</style>
<h1>Testing st.html()</h1>
""")

st.write("## Method 2: st.markdown() with unsafe_allow_html")
st.markdown("""
<style>
button { background: green !important; color: white !important; }
</style>
<button>Test Button</button>
""", unsafe_allow_html=True)

st.write("## Method 3: Direct CSS targeting Streamlit elements")
st.markdown("""
<style>
[data-testid="stMainBlockContainer"] { background: yellow !important; }
</style>
""", unsafe_allow_html=True)

st.button("Normal Button")
