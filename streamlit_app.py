# streamlit_app.py
"""
Streamlit Community Cloud Entry Point.

This file acts as a wrapper pointing to `app.py` to allow seamless deployment
on Streamlit Community Cloud, which defaults to looking for `streamlit_app.py`.
"""

import sys
import os

# Insert the parent directory into python path to ensure imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main dashboard application to run it
import app
