import streamlit as st

from llm_agent import interroger_agent_wordle, load_dictionary


# ------------------------
# Page config
# ------------------------
st.set_page_config(
    page_title="Wordle CSP + LLM",
    page_icon="🟩",
    layout="centered",
)


# ------------------------
# Dictionary + session state
# ------------------------
@st.cache_data
def get_dictionary():
    return load_dictionary("wordle.txt")


DICTIONARY = get_dictionary()

if "attempts" not in st.session_state:
    st.session_state.attempts = []  # [(GUESS, FEEDBACK), ...]
if "history_inputs" not in st.session_state:
    st.session_state.history_inputs = []  # [{"Guess":..., "Feedback":...}, ...]
if "history_prompts" not in st.session_state:
    st.session_state.history_prompts = []  # free-text prompts (optional)
if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ------------------------
# Title
# ------------------------
st.markdown(
    "<h1 style='text-align:center;'>🟩🟨⬜ Wordle Solver (CSP + LLM)</h1>",
    unsafe_allow_html=True,
)
st.divider()


# ------------------------
# Input mode
# ------------------------
mode = st.radio(
    "Input mode",
    ["Wordle (guess + feedback)", "Free text (LLM extraction)"],
    horizontal=True,
)

with st.expander("Help / expected format", expanded=False):
    st.write("Feedback letters: **V=green**, **J=yellow**, **G=gray**.")
    st.write("Recommended input: `ORATE GVVJG` or `ORATE -> GVVJG`.")
    st.write("The solver keeps a session history of attempts; each new attempt is added to the constraints.")


# ------------------------
# User input
# ------------------------
guess = ""
feedback = ""
prompt = ""

if mode == "Wordle (guess + feedback)":
    col1, col2 = st.columns([2, 2])

    with col1:
        guess = st.text_input("Guess (5 letters)", max_chars=5, placeholder="e.g., ORATE")
    with col2:
        feedback = st.text_input("Feedback (V/J/G)", max_chars=5, placeholder="e.g., GVVJG")

    if guess and feedback:
        prompt = f"{guess.strip().upper()} {feedback.strip().upper()}"
        st.caption("Sent to the agent:")
        st.code(prompt)

else:
    prompt = st.text_area(
        "Describe your attempt (must include guess + feedback somewhere)",
        height=120,
        placeholder="Example: ORATE -> GVVJG",
    )


# ------------------------
# Actions
# ------------------------
colA, colB = st.columns([1, 1])
with colA:
    run_now = st.button("Solve", use_container_width=True)
with colB:
    reset_now = st.button("Reset game", use_container_width=True)


if reset_now:
    st.session_state.attempts = []
    st.session_state.history_inputs = []
    st.session_state.history_prompts = []
    st.session_state.last_result = None
    st.success("Reset done.")


# ------------------------
# Solve
# ------------------------
if run_now:
    if not DICTIONARY:
        st.error("Dictionary is empty. Please check wordle.txt.")
    elif not prompt.strip():
        st.error("Please enter an attempt.")
    else:
        with st.spinner("Thinking..."):
            try:
                # Keep a history of free-text prompts (optional)
                if mode == "Free text (LLM extraction)":
                    st.session_state.history_prompts.append(prompt.strip())

                result = interroger_agent_wordle(
                    prompt_utilisateur=prompt,
                    dictionary_words=DICTIONARY,
                    attempts=st.session_state.attempts,
                )
                st.session_state.last_result = result

                # If the user used the structured mode, log it nicely
                if mode == "Wordle (guess + feedback)":
                    g = guess.strip().upper()
                    f = feedback.strip().upper()
                    if len(g) == 5 and len(f) == 5:
                        st.session_state.history_inputs.append({"Guess": g, "Feedback": f})

            except Exception as e:
                st.error(f"Error: {e}")


# ------------------------
# Output
# ------------------------
if st.session_state.last_result:
    st.divider()
    st.subheader("Result")
    st.text(st.session_state.last_result)


# ------------------------
# Current game table
# ------------------------
if st.session_state.history_inputs:
    st.divider()
    st.subheader("Current game (guess + feedback)")
    st.table(st.session_state.history_inputs)


# ------------------------
# Attempts debug (optional but useful)
# ------------------------
with st.expander("Session attempts (debug)", expanded=False):
    st.write(st.session_state.attempts)


# ------------------------
# Free-text history (optional)
# ------------------------
if st.session_state.history_prompts:
    st.divider()
    st.subheader("Free-text prompts history")
    with st.expander("Show history", expanded=False):
        for i, p in enumerate(reversed(st.session_state.history_prompts), 1):
            st.write(f"{i}. {p}")


st.divider()
st.caption("Run with: `streamlit run app.py`")

