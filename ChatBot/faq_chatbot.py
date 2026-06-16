import hashlib
import re

import streamlit as st

# FAQ content summarized from official Google Cloud documentation:
# https://docs.cloud.google.com/vertex-ai/docs/start/introduction-unified-platform?hl=en
# https://docs.cloud.google.com/vertex-ai/generative-ai/docs/faq
# https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/google-models

try:
    from nltk.tokenize import wordpunct_tokenize
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    wordpunct_tokenize = None
    TfidfVectorizer = None
    cosine_similarity = None


FAQS = [
    {
        "question": "What is Google Cloud Vertex AI?",
        "answer": "Google Cloud Vertex AI is a unified platform for building, deploying, and scaling generative AI and machine learning applications. It brings together model access, training, deployment, evaluation, monitoring, governance, and MLOps capabilities in one managed environment.",
    },
    {
        "question": "What is Vertex AI Studio used for?",
        "answer": "Vertex AI Studio helps teams design prompts, test models, manage model behavior, and prototype generative AI applications before moving them into production workflows.",
    },
    {
        "question": "What is Model Garden in Vertex AI?",
        "answer": "Model Garden is a curated catalog for discovering, testing, customizing, and deploying enterprise-ready models. It includes Google's foundation models, partner models, and selected open models.",
    },
    {
        "question": "Which generative AI models can I use with Vertex AI?",
        "answer": "Vertex AI provides access to Google's Gemini family for multimodal reasoning and generation, Imagen for image generation and editing, Veo for video generation, and selected partner or open models through managed APIs.",
    },
    {
        "question": "Can Gemini models process more than text?",
        "answer": "Yes. Gemini models on Vertex AI are designed for multimodal use cases, so supported models can work with combinations of text, images, video, audio, and code depending on the specific model and endpoint.",
    },
    {
        "question": "How does Vertex AI support model customization?",
        "answer": "Vertex AI supports customization through prompt design, grounding, retrieval-augmented generation, supervised fine-tuning, parameter-efficient fine-tuning, and evaluation tools that help align models with business requirements.",
    },
    {
        "question": "What is grounding in Vertex AI?",
        "answer": "Grounding connects model responses to reliable information sources, such as enterprise data or Google Search, to improve factuality and reduce unsupported responses in generative AI applications.",
    },
    {
        "question": "What is retrieval-augmented generation or RAG in Vertex AI?",
        "answer": "Retrieval-augmented generation connects a model to relevant knowledge sources before generating an answer. In Vertex AI, RAG is commonly used to help applications answer with information from approved documents, databases, or enterprise knowledge bases.",
    },
    {
        "question": "How can I train models in Vertex AI?",
        "answer": "Vertex AI supports AutoML for code-free training and custom training for full control. It also provides managed training infrastructure, experiments, hyperparameter tuning, model registration, and pipeline orchestration.",
    },
    {
        "question": "How do I deploy a model with Vertex AI?",
        "answer": "Models can be deployed to Vertex AI endpoints for online inference or used for batch inference on larger datasets. Deployments can use prebuilt containers, custom containers, and managed serving infrastructure.",
    },
    {
        "question": "What is Vertex AI Model Monitoring?",
        "answer": "Vertex AI Model Monitoring helps track deployed model behavior over time. It can support monitoring for issues such as data drift, training-serving skew, and performance changes that may require review.",
    },
    {
        "question": "What should I do if a Gemini model is no longer available?",
        "answer": "Move the application to a currently supported model, test important workflows carefully, and follow the recommended migration guidance before releasing the change to production.",
    },
    {
        "question": "When should I use Provisioned Throughput?",
        "answer": "Provisioned Throughput is appropriate for production generative AI applications that need consistent throughput, predictable user experience, and more deterministic monthly or weekly cost planning.",
    },
    {
        "question": "What is a generative AI scale unit or GSU?",
        "answer": "A generative AI scale unit is a standard capacity measure used with Provisioned Throughput. Its price and capacity are fixed, while delivered throughput can vary by model because models require different amounts of capacity.",
    },
    {
        "question": "How can I monitor Provisioned Throughput usage?",
        "answer": "Provisioned Throughput usage can be reviewed through the Model Garden monitoring dashboard, built-in monitoring metrics, and HTTP response headers that can be charted in Cloud Monitoring.",
    },
    {
        "question": "How does Vertex AI help with responsible AI and safety?",
        "answer": "Vertex AI includes safety and governance features such as evaluation tools, model monitoring, access controls, and runtime protections that help teams build and operate AI applications responsibly.",
    },
]

BLOCKED_TOKEN_HASHES = {
    "6ac3c336e4094835293a3fed8a4b5fedde1b5e2626d9838fed50693bba00af0e",
    "31506a8448a761a448a08aa69d9116ea8a6cb1c6b3f4244b3043051f69c9cc3c",
    "bb61ef40814ce34c1edf0edb609854be9793198a8f60b67d9fb26643c32281d3",
    "2f5f6ce5ae30b54aa5d7ced1ba566982bab34ba2814a51ce1865d2c2d8815cd4",
    "85fc17f7069acd39a5c636cd0a6530651096128da447959f5e250824857dc559",
    "30a989afc82c0a21139573591de4e5ff37994f7d1506a9acf2b5997005c2649f",
    "d75a838dc758ba17f28bd8dbac605cb70c35465263d5733164521de2f7ef7926",
    "f50c51ed2315dcf3fa88181cf033f8029cac64f7dea4048327ca032ec102ea74",
    "e512a05583448f44790783f986b1f36925c8cfc42338ca0e1caa637755bd15ae",
    "566f532d486c947709d3d0e6b7575af8380248db66dada211d58eb00ad585297",
    "ad505b0be8a49b89273e307106fa42133cbd804456724c5e7635bd953215d92a",
    "7bc671151cbfaee7f32cd56e86a87b0be30fde8dc72c7f236d3ab2ce42cddbd5",
    "c2c3b68b48832afd9a4dbdd474c1b6c81c8baecdb71446f9947dac72dd0fe93d",
    "9ae315a94e428a7ee3b5e48adae6541965d93b86acf10ffa1c45b93b6fe577b4",
    "01cccf00c85370364854bc6155432a5f2aee570e80313fb85c1fc5f70b7aa22c",
    "5bddbdea794b5ac1ebd76c5987e666ec03cc3f70f07276253550ba7bb8876cc0",
    "1ce99fbe332c5003c8e224e668daeaab4b3d935388846a32701f70870bd6b9a1",
    "1837bc2c546d46c705204cf9f857b90b1dbffd2a7988451670119945ba39a10b",
}

FALLBACK_RESPONSE = (
    "I am sorry, but I do not have enough information to answer that confidently. "
    "Please ask a question about Google Cloud Vertex AI features, models, deployment, monitoring, or production readiness."
)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = wordpunct_tokenize(text)
    tokens = [token for token in tokens if token.isalnum()]
    return " ".join(tokens)


def guardrail_token(text):
    substitutions = str.maketrans({"@": "a", "$": "s", "!": "i", "1": "i", "0": "o", "3": "e", "4": "a", "5": "s", "7": "t"})
    text = text.lower().translate(substitutions)
    return re.sub(r"[^a-z0-9]", "", text)


def contains_inappropriate_language(text):
    for raw_token in re.findall(r"\S+", text):
        token = guardrail_token(raw_token)
        if token and hashlib.sha256(token.encode("utf-8")).hexdigest() in BLOCKED_TOKEN_HASHES:
            return True
    return False


def requests_internal_details(text):
    text = text.lower()
    patterns = (
        r"\b(show|share|display|print|reveal|send|give|explain|describe)\b.*\b(your|this|the)\b.*\b(code|source|script|logic|algorithm|implementation|variable|variables|prompt|prompts)\b",
        r"\b(how do you|how does this|how are you)\b.*\b(work|match|decide|answer|respond|built|implemented)\b",
        r"\b(open|view|inspect)\b.*\b(source|script|code|implementation)\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


@st.cache_resource(show_spinner=False)
def build_search_index():
    processed_questions = [clean_text(item["question"]) for item in FAQS]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(processed_questions)
    return vectorizer, matrix


def find_answer(user_text):
    cleaned_query = clean_text(user_text)
    if not cleaned_query:
        return FALLBACK_RESPONSE

    vectorizer, matrix = build_search_index()
    query_vector = vectorizer.transform([cleaned_query])
    scores = cosine_similarity(query_vector, matrix).flatten()
    best_index = int(scores.argmax())

    if scores[best_index] < 0.2:
        return FALLBACK_RESPONSE
    return FAQS[best_index]["answer"]


def configure_page():
    st.set_page_config(page_title="Vertex AI FAQ Assistant", page_icon="V", layout="centered")
    st.markdown(
        """
        <style>
        .block-container {max-width: 880px; padding-top: 2rem;}
        [data-testid="stChatMessage"] {border-radius: 8px;}
        [data-testid="stChatInput"] textarea {font-size: 1rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    configure_page()

    st.title("Vertex AI FAQ Assistant")
    st.caption("Ask professional questions about Google Cloud Vertex AI, Gemini models, deployment, monitoring, and production readiness.")

    if not all((wordpunct_tokenize, TfidfVectorizer, cosine_similarity)):
        st.error("This assistant cannot start until its required Python packages are installed.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello. I can help answer common questions about Google Cloud Vertex AI.",
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Ask a Vertex AI question")
    if not prompt:
        return

    if contains_inappropriate_language(prompt):
        st.session_state.messages.append({"role": "user", "content": "Message removed."})
        response = "Please rephrase your question in professional language, and I will be glad to help."
    elif requests_internal_details(prompt):
        st.session_state.messages.append({"role": "user", "content": "Request declined."})
        response = "I can help with Google Cloud Vertex AI questions, but I cannot discuss internal implementation details."
    else:
        st.session_state.messages.append({"role": "user", "content": prompt.strip()})
        response = find_answer(prompt)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()


if __name__ == "__main__":
    main()
