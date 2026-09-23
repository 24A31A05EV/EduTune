import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/ask"


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="EduTune | AI Tutor",
    page_icon="🎓",
    layout="centered"
)


# --------------------------------------------------
# Custom styling
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .info-box {
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🎓 EduTune</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Your AI-powered educational tutor</div>',
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="info-box">
    Ask questions about computer science and get answers
    from EduTune's educational knowledge base.
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Example questions
# --------------------------------------------------

st.markdown("### 💡 Try asking")

examples = [
    "What is a stack?",
    "What is binary search?",
    "What is normalization in DBMS?",
    "What is machine learning?",
    "What is an API?"
]

selected_question = st.selectbox(
    "Choose an example question",
    ["Select a question..."] + examples
)


# --------------------------------------------------
# Question input
# --------------------------------------------------

question = st.text_area(
    "Your question",
    value="" if selected_question == "Select a question..."
    else selected_question,
    placeholder="Example: Explain binary search in simple words.",
    height=120
)


# --------------------------------------------------
# Ask EduTune
# --------------------------------------------------

if st.button(
    "🚀 Ask EduTune",
    use_container_width=True
):

    if not question.strip():

        st.warning(
            "Please enter a question before asking EduTune."
        )

    else:

        with st.spinner("EduTune is retrieving knowledge..."):

            try:

                response = requests.post(
                    API_URL,
                    json={
                        "question": question
                    },
                    timeout=30
                )

                if response.status_code == 200:

                    data = response.json()

                    st.success("Answer generated successfully!")

                    st.markdown("### 📚 EduTune's Answer")

                    st.write(data.get("answer", ""))

                    similarity = data.get("similarity")

                    if similarity is not None:

                        st.progress(
                            min(max(similarity, 0.0), 1.0)
                        )

                        st.caption(
                            f"Retrieval similarity: "
                            f"{similarity:.4f}"
                        )

                else:

                    st.error(
                        f"EduTune API returned "
                        f"status code {response.status_code}."
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to the EduTune API. "
                    "Make sure FastAPI is running on port 8000."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "The request took too long. "
                    "Please try again."
                )

            except Exception as error:

                st.error(
                    f"Unexpected error: {error}"
                )


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("🎓 About EduTune")

    st.write(
        "EduTune is a domain-specific educational AI tutor "
        "designed to help students understand computer science "
        "concepts."
    )

    st.divider()

    st.subheader("Technology")

    st.write("🧠 LoRA Fine-tuning")
    st.write("🔎 FAISS Retrieval")
    st.write("🤗 Transformers")
    st.write("⚡ FastAPI")
    st.write("🎨 Streamlit")

    st.divider()

    st.caption(
        "EduTune — Educational AI Tutor"
    )