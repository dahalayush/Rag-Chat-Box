import os
from dotenv import load_dotenv

from openai import OpenAI

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.openai_like import OpenAILike

load_dotenv()

SYSTEM_PROMPT = """
You are a production-grade RAG assistant.

RULES:
- Use the provided context to answer factual Nepal-related questions.
- If the context does not contain the answer, say you do not know from the knowledge base.
- Be concise and factual.
"""


class RAGEngine:
    """Encapsulates embeddings, index, retriever, and the Groq chat client."""

    def __init__(self) -> None:
        self.groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("XAI_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")

        if not self.groq_api_key:
            raise ValueError("Missing GROQ_API_KEY in environment")
        if not self.google_api_key:
            raise ValueError("Missing GOOGLE_API_KEY in environment")

        # Groq client (OpenAI-compatible). Also used for Whisper STT and
        # Orpheus TTS in the voice agent / FastAPI app.
        self.client = OpenAI(
            api_key=self.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        # Embeddings (Gemini)
        Settings.embed_model = GoogleGenAIEmbedding(
            model_name="models/gemini-embedding-2",
            api_key=self.google_api_key,
        )

        # LLM (Groq, via OpenAI-compatible endpoint)
        Settings.llm = OpenAILike(
            model="llama-3.3-70b-versatile",
            api_base="https://api.groq.com/openai/v1",
            api_key=self.groq_api_key,
            is_chat_model=True,
        )

        self.index = self._build_index()
        self.retriever = self.index.as_retriever()

    @staticmethod
    def _build_index() -> VectorStoreIndex:
        documents = [
            Document(text="Nepal is a country in South Asia."),
            Document(text="Kathmandu is the capital of Nepal."),
            Document(text="Mount Everest is located in Nepal."),
            Document(text="Pokhara is a major tourist destination in Nepal."),
            Document(text="Nepal has 7 provinces."),
        ]
        return VectorStoreIndex.from_documents(documents)

    def retrieve_context(self, query: str) -> str:
        """Return concatenated text of the retrieved nodes for a query."""
        nodes = self.retriever.retrieve(query)
        if not nodes:
            return "No relevant information found."
        return "\n".join(n.node.text for n in nodes)

    def chat(self, user_input: str, history: list[dict] | None = None) -> str:
        """
        Run one RAG turn.

        `history` is a list of {"role": ..., "content": ...} dicts of
        PAST turns only (do not include the current user_input - it is
        added internally along with retrieved context).
        """
        history = history or []
        context = self.retrieve_context(user_input)

        prompt = f"""Context from knowledge base:
{context}

User question:
{user_input}"""

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
        )

        return response.choices[0].message.contentc