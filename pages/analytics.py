# pages/analytics.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from modules.auth import require_auth

st.set_page_config(page_title="Аналитика", layout="wide")

require_auth()

st.title("Аналитика")
st.write("Раздел в разработке.")
