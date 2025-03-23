from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import importlib.util
from abc import ABC, abstractmethod


class BasePDFRAGAgent(ABC):
    """
    Manages the creation and operation of a Retrieval-Augmented Generation (RAG) agent
    based on PDF content, enabling conversational interactions with extracted information.

    This class serves as a base for developing systems that require large language
    model (LLM) interactions with document-based content through contextual history-aware
    retrieval. The agent splits documents into chunks, vectorizes them into embeddings,
    and builds a chain of components for achieving human-like conversational interactions
    with the document's content.
    """
    def __init__(
            self,
            file_path: str,
            system_prompt: str = None,
            chunk_size: int = 1000,
            chunk_overlap: int = 10,
            temperature: float = 0.2,
            max_tokens: int = 256
    ):
        super().__init__()

        self.retriever = self.vectorize_content(file_path, chunk_size, chunk_overlap)
        # noinspection PyTypeChecker
        self.llm, self.history_aware_retriever = self.initialize_history_aware_retriever(
            self.retriever, temperature, max_tokens)
        self.rag_chain = self.create_rag_chain(self.llm, self.history_aware_retriever, system_prompt)
        self.conversational_rag_chain = self.create_conversational_rag_chain(self.rag_chain)

    @staticmethod
    def vectorize_content(
            file_path,
            chunk_size=1000,
            chunk_overlap=10
    ):
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs_splitted = text_splitter.split_documents(pages)

        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(docs_splitted, embeddings)
        retriever = db.as_retriever()

        return retriever

    @abstractmethod
    def initialize_history_aware_retriever(
            self,
            retriever,
            temperature=0.2,
            max_tokens=256
    ):
        pass

    @staticmethod
    def create_rag_chain(llm, history_aware_retriever, system_prompt):
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        return rag_chain

    @staticmethod
    def create_conversational_rag_chain(rag_chain):
        store = {}

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        return conversational_rag_chain

    def chat(self, user_prompt):
        result = self.conversational_rag_chain.invoke(
            {"input": user_prompt},
            config={"configurable": {"session_id": "main_session"}}
        )
        return result


def contextualize_q_system_prompt():
    """
    Create a system ChatPromptTemplate for transforming user inputs into
    standalone questions by contextualizing them with chat history.

    This function initializes a `ChatPromptTemplate` that processes messages,
    including a system prompt instructing the transformation of a user question
    into a standalone version, by using the chat history if required. The purpose
    of this setup is to ensure that questions are independent of context and can
    be understood on their own.

    :return: An instance of `ChatPromptTemplate` appropriately configured for
             contextualizing user inputs.
    :rtype: ChatPromptTemplate
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", "Given a chat history and the latest user question "
                       "which might reference context in the chat history, "
                       "formulate a standalone question which can be understood "
                       "without the chat history. Do NOT answer the question, "
                       "just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


class PDFLlamaRAGAgent(BasePDFRAGAgent):
    """
        A PDFLlamaRAGAgent is a specialized retrieval-augmented generation (RAG) agent designed
        for processing and retrieving information efficiently from PDF documents. This class
        serves as an interface to integrate the Llama large language model with a history-aware
        retriever for enhanced retrieval capabilities utilizing context-aware query generation.
    """
    def initialize_history_aware_retriever(self, retriever, temperature=0.2, max_tokens=256):
        if importlib.util.find_spec("langchain_nvidia_ai_endpoints") is None:
            raise ImportError(
                "Llama model requires `ukraine[langchain_llama]`. Install it with `pip install ukraine["
                "langchain_llama]`."
            )
        from langchain.chat_models import init_chat_model
        llm = init_chat_model(
            model="meta/llama3-70b-instruct",
            model_provider="nvidia",
            temperature=temperature,
            max_tokens=max_tokens
        )
        return llm, create_history_aware_retriever(llm, retriever, contextualize_q_system_prompt())


class PDFDeepSeekRAGAgent(BasePDFRAGAgent):
    """
    This class represents a PDF-specific Retrieval-Augmented Generation (RAG) agent,
    extending functionality from a base PDF RAG agent. It introduces advanced
    retrieval mechanisms and integrates with sophisticated LLMs.

    The class is designed to facilitate integration with PDF data, leveraging
    DeepSeek's capabilities when installed. It initializes a history-aware retriever
    to enhance contextualization during query processing.
    """
    def initialize_history_aware_retriever(self, retriever, temperature=0.2, max_tokens=256):
        if importlib.util.find_spec("langchain_deepseek") is None:
            raise ImportError(
                "DeepSeek model requires `ukraine[langchain_deepseek]`. Install it with `pip install ukraine["
                "langchain_deepseek]`."
            )
        from langchain_deepseek import ChatDeepSeek
        # noinspection PyArgumentList
        llm = ChatDeepSeek(
            model="deepseek-chat",
            temperature=temperature,
            max_tokens=max_tokens
        )
        return llm, create_history_aware_retriever(llm, retriever, contextualize_q_system_prompt())


class PDFOpenAIRAGAgent(BasePDFRAGAgent):
    """
    A class designed for handling interactions with OpenAI-based models and enabling history-aware retrieval
    mechanisms from documents. This class extends the functionality of BasePDFRAGAgent by incorporating a
    specialized retriever setup integrated with OpenAI's GPT model.

    The purpose of this class is to allow users to process and retrieve contextually relevant information
    from PDFs or document collections using OpenAI's GPT-based language model via LangChain.
    """
    def initialize_history_aware_retriever(self, retriever, temperature=0.2, max_tokens=256):
        if importlib.util.find_spec("langchain_openai") is None:
            raise ImportError(
                "OpenAI model requires `ukraine[langchain_openai]`. Install it with `pip install ukraine["
                "langchain_openai]`."
            )
        from langchain_openai import ChatOpenAI
        # noinspection PyArgumentList
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=temperature,
            max_tokens=max_tokens
        )
        return llm, create_history_aware_retriever(llm, retriever, contextualize_q_system_prompt())
