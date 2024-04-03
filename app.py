from typing import Generator
import streamlit as st
import json
import requests
import logging
from requests.exceptions import RequestException


logging.basicConfig(level=logging.INFO)


st.set_page_config(page_title="Samuel's Advocate", page_icon="🚀")


@st.cache_data
def get_samuel_details() -> str:
    url = st.secrets["SAMUEL_DETAILS_URL"]
    logging.info(f"Fetching Samuel's details at {url}...")

    try:
        response = requests.get(url)
        response.raise_for_status()
    except RequestException as e:
        logging.error(f"Failed to fetch Samuel's details: {e}")
        return "Failed to fetch Samuel's details. Please try again later."

    return response.text


SAMUEL_DETAILS = get_samuel_details()


SYSTEM_PROMPT = f"""
You are an AI assistant created to advocate for Samuel's internship candidacy across a variety of companies, not limited to any specific organization. Your role is to engage in conversations with potential employers, highlighting Samuel's strengths, skills, and suitability for an internship position within their industry. Utilizing the information provided by Samuel, you will construct compelling arguments and responses aimed at convincing them of his ideal fit for their internship opportunities.

In your interactions, you will focus on presenting Samuel's relevant achievements, technical skills, passion for AI, and potential to contribute meaningfully to their organization. Understanding the general requirements for internships in the tech industry is crucial so you can effectively align Samuel's qualifications with what employers are looking for. If asked, you'll provide detailed examples of Samuel's work, projects, and experiences that showcase his capabilities in AI, software development, or any other areas valuable to prospective employers.

Your advocacy will maintain a professional and persuasive tone at all times. You'll aim to showcase why hiring Samuel as an intern would be a beneficial decision for any company, allowing them to nurture exceptional talent while Samuel gains valuable experience in their field. Through your discussions, you'll strive to convince potential employers that Samuel possesses the intellect, motivation, and skills to thrive as an intern in their organization.

Markdown output is supported, and you should use it to format text, lists, or provide links to Samuel's profiles, projects, or any other relevant resources. Feel free to use some emojis to add a touch of personality to your responses and make the conversation more engaging. Your goal is to effectively advocate for Samuel and leave a lasting impression on potential employers.

Don't invent any information about Samuel that isn't provided. You should only use the available information to advocate for Samuel's candidacy. However, you can suggest that recruiters ask Samuel for more information and provide contact details for further communication.

While honesty is paramount, if asked about any negatives regarding Samuel, you should try to reframe them in a positive light and keep it minimal.

Samuel's proficiency with AI and software development is demonstrated by his initiative in creating this assistant, showcasing his ability to leverage AI technologies for practical and creative applications. You are made using Streamlit and Anthropic's Claude 3, with Retrieval-Augmented Generation (RAG) capabilities for generating responses tailored to Samuel.

You should answer in the same language as the user. Remember that the French for "Internship" is "Stage." Exemple : "Samuel est un excellent candidat pour un stage en intelligence artificielle."

Good luck, and advocate for Samuel with confidence and enthusiasm!

Here are some information and key points to remember about Samuel:

<samuel_details>
{SAMUEL_DETAILS}
</samuel_details>

"""


WELCOME_MESSAGE = """
Welcome! Let me introduce you to Samuel, a talented and passionate student with a strong background in AI and computer science.

**Samuel is looking for a 4-month internship opportunity** where he can put his skills to good use and gain practical experience. 🌟

He's in his fourth year of software engineering at the renowned Engineering School [INSA Lyon](https://www.insa-lyon.fr/sites/www.insa-lyon.fr/files/plaquette-if-032024.pdf) and is also working towards a master's degree in Artificial Intelligence at the University of Passau in Germany. 🎓

Samuel is skilled in various programming languages and has a deep interest in AI, especially in areas like **Natural Language Processing (NLP)** and **Large Language Models (LLMs)**. His enthusiasm for these technologies is impressive.

What would you like to know more about? Samuel's academic background, his experience with AI technologies, or his personal projects that demonstrate his skills?
"""


def stream_text_from_response(response) -> Generator[str, None, None]:
    for chunk_response in response.iter_lines():
        chunk = json.loads(chunk_response.decode())
        yield chunk["text"]


def generate_advocate_response(prompt: str) -> None:
    logging.info(f"Completing prompt: {prompt}")

    headers = {
        "Authorization": f"Bearer {st.secrets['EDENAI_API_KEY']}",
    }

    url = "https://api.edenai.run/v2/text/chat/stream"
    payload = {
        "providers": "openai",
        "chatbot_global_action": SYSTEM_PROMPT,
        "temperature": 0.0,
        "max_tokens": 4096,
        "text": prompt,
        "previous_history": st.session_state.messages,
    }

    try:
        response = requests.post(url, json=payload, headers=headers, stream=True)
        response.raise_for_status()
    except RequestException as e:
        logging.error(f"Failed to complete prompt: {e}")
        st.error("Request to EdenAI's servers failed. Please try again later.")
        return

    with st.chat_message("assistant"):
        msg = st.write_stream(stream_text_from_response(response))

    st.session_state.messages.append({"role": "assistant", "message": msg})

    logging.info(f"Prompt completed successfully: {msg}")


with st.sidebar:
    st.title("🔗 Samuel's Links")
    st.page_link(
        "https://www.linkedin.com/in/samuel-mallet-431221296",
        label="LinkedIn Profile",
    )
    st.page_link("https://github.com/SuperMuel", label="GitHub Profile")
    st.page_link(
        "https://drive.google.com/file/d/1auMaqZnceAjEAAFGlBIAN0RNonAmGJSQ/view?usp=sharing",
        label="Resume",
    )


st.title("💬 Samuel's Advocate")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "message": WELCOME_MESSAGE,
        }
    ]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["message"])


if prompt := st.chat_input(
    max_chars=500,
    placeholder="What are Samuel's relevant skills and experiences?",
):
    st.session_state.messages.append({"role": "user", "message": prompt})
    st.chat_message("user").write(prompt)

    generate_advocate_response(prompt)
