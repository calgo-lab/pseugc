import html
import json
import requests
import streamlit as st
import time
import random

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

# Example texts to choose from
example_texts = [
    "Herr Markus Schneider, geboren am 14. März 1980 und wohnhaft in der Hauptstraße 12 in 10115 Berlin, stellte sich am 22. Mai 2024 mit thorakalen Schmerzen in unserer Notaufnahme vor. Die Beschwerden hatten laut eigener Angabe bereits am 20. Mai begonnen. Es wurde ein akutes Koronarsyndrom in Kombination mit einer bekannten Hypertonie diagnostiziert. Die Therapie bestand aus einer sofortigen Gabe von Acetylsalicylsäure und Heparin, woraufhin eine stationäre Aufnahme erfolgte. Am 23. Mai wurde eine koronare Angiographie durchgeführt. Der Patient konnte am 27. Mai 2024 in stabilem Zustand entlassen und zur weiteren Betreuung an seinen Hausarzt, Dr. Julia Meier in der Karl-Marx-Allee 50, übergeben werden.",
    "Frau Sabrina Koch, geboren am 2. November 1972, wohnhaft im Lindenweg 8, 80331 München, berichtete über eine zunehmende Belastungsdyspnoe, die sich über mehrere Wochen entwickelt hatte. In der internistischen Abklärung wurde eine chronisch obstruktive Lungenerkrankung (COPD) im Stadium II festgestellt. Therapeutisch wurde eine bronchodilatatorische Inhalationstherapie mit Salbutamol sowie eine kurzfristige Prednisolon-Gabe eingeleitet. Nach einer klinischen Stabilisierung wurde Frau Koch am 10. April 2025 entlassen und gebeten, sich bei ihrem Lungenfacharzt, Dr. Anton Berg in der Leopoldstraße 75, zur Verlaufskontrolle vorzustellen.",
    "Herr Peter Brandt, geb. 8. August 1955, wohnhaft in der Schulstraße 23, 04109 Leipzig, suchte unsere Einrichtung aufgrund seit Monaten bestehender Rückenschmerzen auf. Eine bildgebende Diagnostik bestätigte einen Bandscheibenvorfall auf Höhe L4/L5. Es zeigten sich jedoch keine neurologischen Defizite. Wir empfahlen zunächst eine konservative Therapie bestehend aus gezielter Physiotherapie und analgetischer Medikation. Nach ambulanter Vorstellung und ausführlicher Beratung wurde Herr Brandt am 17. Juni 2025 zur weiteren Behandlung an Dr. Yvonne Schröder vom Orthopädiezentrum Leipzig überwiesen.",
    "Frau Dr. med. Hannah Reuter, geboren am 29. Juni 1975, gesetzlich versichert bei der AOK (Versichertennummer 123456789), wurde am 3. Februar 2025 wegen einer mittelgradigen depressiven Episode (ICD-10: F32.1) stationär aufgenommen. Die Patientin klagte über Antriebslosigkeit, Schlafstörungen und gedrückte Stimmungslage. Im Rahmen des Aufenthalts wurde eine antidepressive Medikation mit Sertralin eingeleitet, begleitet von kognitiver Verhaltenstherapie in Einzel- und Gruppensitzungen. Am 17. März 2025 konnte Frau Reuter in gebessertem Zustand entlassen werden. Für die Weiterbehandlung empfehlen wir eine ambulante Psychotherapie bei Herrn Dipl.-Psych. Felix Bauer in Berlin.",
    "Herr Mehmet Yildirim, geboren am 12. Januar 1984 und wohnhaft in der Sonnenallee 104, 12045 Berlin, wurde am 15. Januar 2025 aufgrund einer manifesten Alkoholabhängigkeit stationär aufgenommen. Die Entzugsbehandlung erfolgte unter engmaschiger medizinischer Überwachung. Nach erfolgreichem körperlichem Entzug und begleitender psychotherapeutischer Intervention konnte Herr Yildirim am 30. Januar 2025 entlassen werden. Als Nachsorge wird die Teilnahme an der Selbsthilfegruppe „Nüchtern leben“ in Berlin-Neukölln dringend empfohlen.",
    "Frau Anna-Lena Weiß, geboren am 19. Juli 2002, wohnhaft in der Mozartstraße 9, 68161 Mannheim, wurde am 5. April 2025 stationär in unserer psychosomatischen Klinik aufgenommen. Die Aufnahme erfolgte aufgrund einer generalisierten Angststörung (ICD-10: F41.1), die sich in dauerhafter innerer Unruhe, Konzentrationsproblemen und körperlichen Symptomen äußerte. Therapeutisch kamen sowohl Verhaltenstherapie als auch Atemtechniken und eine medikamentöse Behandlung mit Escitalopram zum Einsatz. Nach zwei Wochen stabiler Verbesserung wurde sie am 19. April 2025 entlassen und zur weiteren Behandlung an Frau Dr. Catharina Lenz vom Mannheimer Zentrum für Angststörungen übergeben.",
    "Tim-Oliver Neumann, geboren am 12. September 2010, wurde am 18. Januar 2025 im Universitätsklinikum Leipzig aufgrund einer akuten Appendizitis operativ behandelt. Die Entscheidung zur sofortigen Operation erfolgte nach positiver klinischer Untersuchung und laborchemischem Nachweis einer Entzündung. Die Appendektomie wurde minimal-invasiv durchgeführt und verlief komplikationslos. Der Patient konnte am 21. Januar 2025 in gutem Allgemeinzustand entlassen werden.",
    "Frau Heike Möller, geboren am 25. April 1963, stellte sich im Helios Klinikum München West mit rechtsseitigen Oberbauchschmerzen vor. Die Diagnostik bestätigte das Vorliegen multipler Gallensteine mit wiederholten Koliken. Am 2. Mai 2025 wurde eine laparoskopische Cholezystektomie durchgeführt. Der postoperative Verlauf gestaltete sich unauffällig. Frau Möller wurde am 5. Mai 2025 in gutem Zustand entlassen mit der Empfehlung zur Nachkontrolle bei ihrem Hausarzt, Dr. Stefan Knoll.",
    "Dr. Thomas Henke, geboren am 1. Januar 1970, wurde am 10. März 2025 in der neurochirurgischen Abteilung der Charité Berlin aufgrund eines Bandscheibenprolapses im Segment L5/S1 operiert. Die mikrochirurgische Diskektomie verlief ohne Komplikationen. Bereits am ersten postoperativen Tag zeigte sich eine deutliche Besserung der Beinschmerzen. Am 14. März 2025 konnte Herr Dr. Henke beschwerdearm nach Hause entlassen werden.",
]

# Initialize session state variable if not set
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# App title
st.title("Text Pseudonymization App")


# Button to choose a random example
if st.button("Use Example Text"):
    st.session_state.input_text = random.choice(example_texts)

# Input text area using the session state variable
input_text_area = st.text_area("Enter text here:", value=st.session_state.input_text, height=150)


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
                decorated_output += html.escape(output_text[prev_end: start_idx])

                # Add and decorate the original token with strikethrough
                decorated_output += (
                    f'<span class="{label.lower()}-label label-extra label-token">{html.escape(token)}</span> '
                )

                # Place and decorate the pseudonym
                decorated_output += (
                    f'<span class="{label.lower()}-label label-extra">{html.escape(pseudonym)}</span>'
                )

                # Update last processed index
                prev_end = start_idx + len(pseudonym)

        # Add remaining text
        decorated_output += html.escape(output_text[prev_end:])
        
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
