#!/bin/bash

# Start the Python app (Flask)
python3 filewatcher.py &

# Start the Streamlit app
streamlit run resume_screener.py &

# Wait to keep the container alive
wait
