import html
import random
import requests
import streamlit as st

# Model API URL
# API_URL = "http://pseugc-app.project.ris.bht-berlin.de/predict"
API_URL = "http://localhost:8000/predict"

CODEALLTAG_LABELS = ["CITY", "DATE", "EMAIL", "FAMILY", "FEMALE", "MALE", "ORG", 
                     "PHONE", "STREET", "STREETNO", "UFID", "URL", "USER", 
                     "ZIP"]

LER_LABELS = ["CITY", "DATE", "EMAIL", "FAMILY", "FEMALE", "MALE", "ORG"]

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

    "Dr. Thomas Henke, geboren am 1. Januar 1970, wurde am 10. März 2025 in der neurochirurgischen Abteilung der Charité Berlin aufgrund eines Bandscheibenprolapses im Segment L5/S1 operiert. Die mikrochirurgische Diskektomie verlief ohne Komplikationen. Bereits am ersten postoperativen Tag zeigte sich eine deutliche Besserung der Beinschmerzen. Am 14. März 2025 konnte Herr Dr. Henke beschwerdearm nach Hause entlassen werden."
]

# App page title and favicon
st.set_page_config(page_title="Redakto", page_icon="favicon.ico")

if "entity_set_id" not in st.session_state:
    st.session_state["entity_set_id"] = "codealltag"
if "model_id" not in st.session_state:
    st.session_state["model_id"] = "mt5"
if "repeat" not in st.session_state:
    st.session_state["repeat"] = 1
if "processed_data" not in st.session_state:
    st.session_state["processed_data"] = None
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

# Custom CSS listing
st.markdown(
    """
    <style>
    :root {
        --decorated-output-bg: white;
        --decorated-output-text: black;
        --code-color: black;

        --city-bg: #B388FF;      --city-border: #7C4DFF;
        --date-bg: #FF8A80;      --date-border: #FF5252;
        --email-bg: #F3E5F5;     --email-border: #E1BEE7;
        --family-bg: #EEFF41;    --family-border: #C6FF00;
        --female-bg: #B2FF59;    --female-border: #76FF03;
        --male-bg: #69F0AE;      --male-border: #00E676;
        --org-bg: #FFB74D;       --org-border: #FFA726;
        --phone-bg: #FF99FF;     --phone-border: #CC7ACC;
        --street-bg: #42A5F5;    --street-border: #2196F3;
        --streetno-bg: #81D4FA;  --streetno-border: #4FC3F7;
        --ufid-bg: #D2B48C;      --ufid-border: #A89070;
        --url-bg: #FFEA00;       --url-border: #FFD600;
        --user-bg: #E6E6A3;      --user-border: #B8B882;
        --zip-bg: #B2DFDB;       --zip-border: #80CBC4;
    }

    /* Browser/OS dark mode detection */
    @media (prefers-color-scheme: dark) {
        :root {
            --decorated-output-bg: black;
            --decorated-output-text: white;
            --code-color: wheat;
        }
    }

    body.theme-dark {
        --decorated-output-bg: black;
        --decorated-output-text: white;
        --code-color: wheat;
    }

    .city-label      { background-color: var(--city-bg);      border: 2px solid var(--city-border); }
    .date-label      { background-color: var(--date-bg);      border: 2px solid var(--date-border); }
    .email-label     { background-color: var(--email-bg);     border: 2px solid var(--email-border); }
    .family-label    { background-color: var(--family-bg);    border: 2px solid var(--family-border); }
    .female-label    { background-color: var(--female-bg);    border: 2px solid var(--female-border); }
    .male-label      { background-color: var(--male-bg);      border: 2px solid var(--male-border); }
    .org-label       { background-color: var(--org-bg);       border: 2px solid var(--org-border); }
    .phone-label     { background-color: var(--phone-bg);     border: 2px solid var(--phone-border); }
    .street-label    { background-color: var(--street-bg);    border: 2px solid var(--street-border); }
    .streetno-label  { background-color: var(--streetno-bg);  border: 2px solid var(--streetno-border); }
    .ufid-label      { background-color: var(--ufid-bg);      border: 2px solid var(--ufid-border); }
    .url-label       { background-color: var(--url-bg);       border: 2px solid var(--url-border); }
    .user-label      { background-color: var(--user-bg);      border: 2px solid var(--user-border); }
    .zip-label       { background-color: var(--zip-bg);       border: 2px solid var(--zip-border); }
        
    .label-extra { padding: 2px 6px; border-radius: 5px; color: black;}
        
    .label-token { background-color: wheat !important; text-decoration: line-through; }

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
        background-color: var(--decorated-output-bg);
        color: var(--decorated-output-text);
        line-height: 2.1;
    }
        
    /* Code block features a native copy-to-clipboard functionality */
    div[data-testid="stCode"] pre {
        border: 1px solid #ddd !important;
        font-family: Arial, sans-serif !important; /* Change font */
        font-size: 16px !important; /* Adjust size */
        background-color: transparent !important;
        color: var(--code-color) !important;
    }
        
    hr {
        border: none !important;
        border-top: 2px dashed gray !important; /* Bold dashed line */
        margin: 20px 0 !important; /* Adjust spacing */
        opacity: 1 !important; /* Ensure visibility */
    }

    h1, [data-testid="stMarkdownContainer"] h1 {
        font-size: 1.9rem !important;
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)

logo_col, text_col = st.columns([0.20, 0.80])

with logo_col:
    st.image("logo.png", width=150)
    st.markdown("<h1 style='text-align: center; margin-top: -50px;'>Redakto</h1>", unsafe_allow_html=True)

with text_col:
    st.markdown(
        """
        <div style='display: flex; align-items: center; height: 100%;'>
            <p style='font-size: 1.9rem; margin: 50px;'>German Text Redactor</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# For smooth radio change experience
def update_entity_set_id():
    st.session_state["entity_set_id"] = st.session_state["entity_set_radio"]
    st.session_state["model_id"] = list(get_supported_models(st.session_state["entity_set_id"]).keys())[0]
    st.session_state["processed_data"] = None

# Entity Set selector
entity_set = st.radio(
    "Entity Set:",
    options=["codealltag", "ler"],
    index=["codealltag", "ler"].index(st.session_state["entity_set_id"]),
    format_func=lambda x: {
        "codealltag": "Email/CodEAlltag",
        "ler": "Legal/LER"
    }[x],
    disabled=1,
    horizontal=True,
    key="entity_set_radio",
    on_change=update_entity_set_id,
)

def update_model_id():
    st.session_state["model_id"] = st.session_state["supported_models_radio"]
    st.session_state["processed_data"] = None

codealltag_supported_models = {
    "bilstmcrf": "A/BiLSTM-CRF",
    "gelectra": "A/GELECTRA",
    "mt5": "P/mT5"
}

ler_supported_models = {
    "bilstmcrf": "A/BiLSTM-CRF",
    "gbert": "A/GermanBERT",
    "xlm-roberta": "A/XLM-RoBERTa",
    "gelectra": "A/GELECTRA"
}

def get_supported_models(entity_set_id):
    if entity_set_id == "codealltag":
        return codealltag_supported_models
    elif entity_set_id == "ler":
        return ler_supported_models
    else:
        return {}

# Model selector
supported_models = st.radio(
    "Supported Models:",
    options=list(get_supported_models(st.session_state["entity_set_id"]).keys()),
    index=list(get_supported_models(st.session_state["entity_set_id"]).keys()).index(st.session_state["model_id"]),
    format_func=lambda x: get_supported_models(st.session_state["entity_set_id"])[x],
    horizontal=True,
    key="supported_models_radio",
    on_change=update_model_id,
)

# Label legend logic
def generate_label_legends(labels):
    return " ".join([
        f'<span class="{label.lower()}-label label-extra">{label}</span>'
        for label in labels
    ])

if st.session_state["entity_set_id"] == "codealltag":
    label_legends = generate_label_legends(CODEALLTAG_LABELS)
elif st.session_state["entity_set_id"] == "ler":
    label_legends = generate_label_legends(LER_LABELS)

st.markdown(
    f'<div class="decorated-output-div">{label_legends}</div>',
    unsafe_allow_html=True
)


# Button to choose a random example
if st.button("Use Example Text"):
    st.session_state.input_text = random.choice(example_texts)
    st.session_state["processed_data"] = None

# Input text area using the session state variable
input_text_area = st.text_area("Enter text here:", value=st.session_state.input_text, height=150)


# Update repeat value based on slider
def update_repeat():
    st.session_state["repeat"] = st.session_state.get("repeat_slider", 1)

# Repeat slider
if st.session_state["entity_set_id"] == "codealltag" and st.session_state["model_id"] == "mt5":
    st.slider(
        "Repeat:",
        min_value=1,
        max_value=5,
        value=st.session_state["repeat"],
        key="repeat_slider",
        on_change=update_repeat
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
                "entity_set_id": st.session_state["entity_set_id"],
                "model_id": st.session_state["model_id"],
                "input_texts": [input_text_area],
                "repeat": st.session_state["repeat"],
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

    output_header = "Pseudonymized Outputs" if st.session_state["model_id"] == "mt5" else f"Annotated Output"
    st.subheader(output_header)

    # API supports multiple text as a list, we process only one text through UI
    output_items = st.session_state["processed_data"]["output"][0]
    
    # Loop through multiple output items (based on repeat slider value)
    for output_idx, output_item in enumerate(output_items):
        
        if st.session_state["model_id"] == "mt5":
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
    
            pseudonym = None
            if "Pseudonym" in output_dict.keys():
                pseudonym = output_dict["Pseudonym"][token_id]
            
            if pseudonym:
                start_idx = output_text.find(pseudonym, prev_end)
            else:
                output_text = input_text_area
                start_idx = output_text.find(token, prev_end)
            
            if start_idx != -1:
                
                # Add text before the found pseudonym/token in pseudonymized output text
                decorated_output += html.escape(output_text[prev_end: start_idx])

                # Add and decorate the original token with strikethrough
                decorated_output += (
                    f'<span class="{label.lower()}-label label-extra label-token">{html.escape(token)}</span> '
                )

                # Place and decorate the pseudonym/label

                if pseudonym:
                    decorated_output += (
                        f'<span class="{label.lower()}-label label-extra">{html.escape(pseudonym)}</span>'
                    )
                else:
                    decorated_output += (
                        f'<span class="{label.lower()}-label label-extra">{html.escape(label)}</span>'
                    )

                # Update last processed index
                if pseudonym:
                    prev_end = start_idx + len(pseudonym)
                else:
                    prev_end = start_idx + len(token)

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

        if pseudonym:
            st.code(body=output_text, wrap_lines=True, language="text")
        
        # Add divider to separate multiple outputs
        st.divider()
