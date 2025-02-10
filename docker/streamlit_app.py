import json
import requests
import streamlit as st
import time

# Model API URL
# API_URL = "http://pseugc-app.project.ris.bht-berlin.de/predict"
API_URL = "http://localhost:8000/predict"

LABELS = ["CITY",
          "DATE",
          "EMAIL",
          "FAMILY",
          "FEMALE",
          "MALE",
          "ORG", 
          "PHONE",
          "STREET",
          "STREETNO",
          "UFID",
          "URL",
          "USER",
          "ZIP"]

# App page title and favicon
st.set_page_config(page_title="Text Pseudonymization App", page_icon="🚀")

# App title
st.title("Text Pseudonymization App")

# Input text area
input_text_area = st.text_area("Enter text here:", height=150)

# Repeat slider
repeat_slider = st.slider("Repeat:", 1, 5, 1)

# Custom CSS listing
st.markdown(
    """
    <style>
        .city-label {
            background-color: #B388FF;
            border: 2px solid #7C4DFF;
        }
        
        .date-label {
            background-color: #FF8A80;
            border: 2px solid #FF5252;
        }
        
        .email-label {
            background-color: #F3E5F5;
            border: 2px solid #E1BEE7;
        }
        
        .family-label {
            background-color: #EEFF41;
            border: 2px solid #C6FF00;
        }
        
        .female-label {
            background-color: #B2FF59;
            border: 2px solid #76FF03;
        }
        
        .male-label {
            background-color: #69F0AE;
            border: 2px solid #00E676;
        }
        
        .org-label {
            background-color: #FFB74D;
            border: 2px solid #FFA726;
        }
        
        .phone-label {
            background-color: #FF99FF;
            border: 2px solid #CC7ACC;
        }
        
        .street-label {
            background-color: #42A5F5;
            border: 2px solid #2196F3;
        }
        
        .streetno-label {
            background-color: #81D4FA;
            border: 2px solid #4FC3F7; 
        }
        
        .ufid-label {
            background-color: #D2B48C;
            border: 2px solid #A89070;
        }
        
        .url-label {
            background-color: #FFEA00;
            border: 2px solid #FFD600;
        }
        
        .user-label {
            background-color: #E6E6A3;
            border: 2px solid #B8B882; 
        }
        
        .zip-label {
            background-color: #B2DFDB;
            border: 2px solid #80CBC4;
        }
        
        .label-extra {
            padding: 2px 6px;
            border-radius: 5px;
        }
        
        .label-token {
            background-color: white !important;
            text-decoration: line-through;
        }

        .circle-number {
            display: inline-block;
            width: 40px;
            height: 40px;
            line-height: 33px;
            text-align: center;
            border-radius: 50%;
            border: 2px solid gray;
            font-size: 20px;
            font-weight: bold;
            color: gray;
            margin: 5px 0px;
        }
    
        .decorated-output-div {
            padding: 10px;
            margin-bottom: 15px;
            border: 1px solid #ddd;
            border-radius: 6px;
            background-color: white;
            line-height: 2.1;
        }
        
        /* Code block features a native copy-to-clipboard functionality */
        div[data-testid="stCode"] pre {
            border: 1px solid #ddd !important;
            font-family: Arial, sans-serif !important; /* Change font */
            font-size: 16px !important; /* Adjust size */
            background-color: transparent !important; /* Remove background color */
            color: black !important; /* Normal text color */
        }
        
        hr {
            border: none !important;
            border-top: 2px dashed gray !important; /* Bold dashed line */
            margin: 20px 0 !important; /* Adjust spacing */
            opacity: 1 !important; /* Ensure visibility */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Label legends div
label_legends = " ".join(
    [
        f'<span class="{label.lower()}-label label-extra">{label}</span>' 
        for label in LABELS
    ]
)
st.markdown(
    f'<div class="decorated-output-div">{label_legends}</div>',
    unsafe_allow_html=True
)


# Session state placeholder for API response
if "processed_data" not in st.session_state:
    st.session_state["processed_data"] = None

# Process button
if st.button("Process"):
    
    # Ensure input is not empty
    if input_text_area.strip():
        
        with st.spinner("Processing..."):
            
            # API request payload
            payload = {
                "input_texts": [input_text_area],
                "repeat": repeat_slider
            }
            
            try:
                # Make the API call
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()  # Raise error if API fails

                # Store response in session state
                st.session_state["processed_data"] = response.json()

            except requests.exceptions.RequestException as request_exception:
                st.error(f"API Error: {request_exception}")
    else:
        st.warning("Please enter text before processing.")

# Display processed output
if st.session_state["processed_data"]:
    
    st.subheader("Pseudonymized Outputs")

    # API supports multiple text as a list, we process only one text through UI
    output_items = st.session_state["processed_data"]["output"][0]
    
    # Loop through multiple output items (based on repeat slider value)
    for output_idx, output_item in enumerate(output_items):
        
        st.markdown(
            f'<div class="circle-number">{output_idx + 1}</div>',
            unsafe_allow_html=True
        )
        
        output_dict = output_item["output_dict"]
        output_text = output_item["output_text"]

        token_ids = output_dict["Token_ID"].keys()
        decorated_output = ""
        # Track last processed index
        prev_end = 0
        
        # Loop through all tokens
        for token_id in token_ids:
            
            label = output_dict["Label"][token_id]
            token = output_dict["Token"][token_id]
            pseudonym = output_dict["Pseudonym"][token_id]

            start_idx = output_text.find(pseudonym, prev_end)
            if start_idx != -1:
                
                # Add text before the found pseudonym in pseudonymized output text
                decorated_output += output_text[prev_end: start_idx]

                # Add and decorate the original token with strikethrough
                decorated_output += (
                    f'<span class="{label.lower()}-label label-extra label-token">{token}</span> '
                )

                # Place and decorate the pseudonym
                decorated_output += (
                    f'<span class="{label.lower()}-label label-extra">{pseudonym}</span>'
                )

                # Update last processed index
                prev_end = start_idx + len(pseudonym)

        # Add remaining text
        decorated_output += output_text[prev_end:]
        
        # Replace all new lines with HTML line break
        decorated_output = decorated_output.replace("\n", "<br>")

        # Display decorated output
        st.markdown(
            f'<div class="decorated-output-div">{decorated_output}</div>',
            unsafe_allow_html=True
        )
        
        # Display plain pseudonymized output in pre formatted block
        st.code(body=output_text, wrap_lines=True, language="text")
        
        # Add divider to separate multiple outputs
        st.divider()
