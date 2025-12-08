import streamlit as st
from grammar_checker import grammar_check
from spelling_checker import spell_check
from tone_checker import tone_check
import base64

# -------------------------
# ADD BACKGROUND IMAGE
# -------------------------

import streamlit as st
# your other imports...

st.set_page_config(layout="wide")

# remove top padding
st.markdown("""
    <style>
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
        }
    </style>
    """, unsafe_allow_html=True)
st.markdown("""
    <style>
        header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

def add_bg_image(image_path):
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Call the background function
add_bg_image("bg.png")   # <-- put your background image file in same folder


# -------------------------
#  LOGIN PAGE FUNCTION
# -------------------------
def login_page():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state["logged_in"] = True
            st.success("Login successful!")
        else:
            st.error("Incorrect username or password")


# -------------------------
#  ANALYZER PAGE FUNCTION
# -------------------------
def analyzer_page():
    st.title("🧞 Genie — Text Analyzer")
    st.write("Grammar Check ✔ | Spell Check ✔ | Tone Detection ✔")

    if "text" not in st.session_state:
        st.session_state["text"] = ""

    user_text = st.text_area("Enter your text here", st.session_state["text"], height=200)

    col1, col2 = st.columns(2)

    # ---- Initialize output variables ----
    grammar_output = ""
    spelling_output = ""
    tone_output = ""

    with col1:
        if st.button("Analyze Text"):
            if user_text.strip() == "":
                st.warning("Please enter some text!")
            else:
                spelling_output = spell_check(user_text)
                grammar_output = grammar_check(spelling_output)
                tone_output = tone_check(grammar_output)

                # Save results to session
                st.session_state["grammar_output"] = grammar_output
                st.session_state["spelling_output"] = spelling_output
                st.session_state["tone_output"] = tone_output

        # Load results if they exist
        spelling_output = st.session_state.get("spelling_output", "")
        grammar_output = st.session_state.get("grammar_output", "")
        tone_output = st.session_state.get("tone_output", "")

        # Only show cards if results are available
        if grammar_output != "":
            card_style = """
            <style>
            .result-card {
                padding: 28px;
                border-radius: 16px;
                background: rgba(255, 255, 255, 0.30);
                backdrop-filter: blur(12px);
                margin-bottom: 22px;
                border-left: 6px solid #4A90E2;
                box-shadow: 0 6px 14px rgba(0,0,0,0.15);
                width: 90%;
                min-height: 180px;
                margin-left: auto;
                margin-right: auto;
            }
            .result-title { font-size: 26px; font-weight: 700; }
            .result-text { font-size: 18px; }
            </style>
            """
            st.markdown(card_style, unsafe_allow_html=True)

            # Grammar
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">Grammar Correction</div>
                    <div class="result-text">{grammar_output}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Spell
            st.markdown(
                f"""
                <div class="result-card" style="border-left:6px solid #E67E22;">
                    <div class="result-title">Spell Correction</div>
                    <div class="result-text">{spelling_output}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Tone
            st.markdown(
                f"""
                <div class="result-card" style="border-left:6px solid #9B59B6;">
                    <div class="result-title">Tone Detection</div>
                    <div class="result-text">{tone_output}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with col2:
        if st.button("Clear Text"):
            st.session_state["text"] = ""
            st.session_state["spelling_output"] = ""
            st.session_state["grammar_output"] = ""
            st.session_state["tone_output"] = ""
            st.experimental_rerun()

# -------------------------
# MAIN APP FLOW
# -------------------------
def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        login_page()
    else:
        analyzer_page()


if __name__ == "__main__":
    main()
