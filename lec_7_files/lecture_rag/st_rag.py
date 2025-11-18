import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

st.title("RAG with Streamlit")


def load_documents_and_get_embeddings():
    """
    Load PDF documents from ./docs, split them into chunks, and build/load a FAISS index.
    """
    loader = DirectoryLoader("docs", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    chunk_size = 1000
    chunk_overlap = 100
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    index_path = "llm_faiss_index"

    if os.path.isdir(index_path):
        # Load existing FAISS index
        docembeddings = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        # Build a new FAISS index and save it
        docembeddings = FAISS.from_documents(texts, embeddings)
        docembeddings.save_local(index_path)

    return docembeddings


def get_chain():
    """
    Build a callable `chain` that mimics the old `load_qa_chain` interface.

    It accepts: {"input_documents": docs, "question": question}
    It returns: {"output_text": answer, "input_documents": docs}
    so the rest of the file (get_answer, etc.) can stay unchanged.
    """
    prompt_template = """
Use the following pieces of context to answer the question at the end.
Make sure the answer leverages the context rather than just your general knowledge.
Before answering, ask yourself if you made meaningful use of the context. If not,
say that it wasn't really covered in the provided material.

Format your response as:

Question: [question here]
Helpful Answer: [answer here]

Begin!

Context:
---------
{context}
---------
Question: {question}
Helpful Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    llm = ChatOpenAI(temperature=0.0)

    def chain(inputs: dict):
        # inputs: {"input_documents": docs, "question": question}
        docs = inputs["input_documents"]
        question = inputs["question"]

        context = "\n\n".join(d.page_content for d in docs)
        prompt_text = prompt.format(context=context, question=question)
        response = llm.invoke(prompt_text)

        # ChatOpenAI returns an object with `.content`
        answer_text = getattr(response, "content", str(response))

        # Match the keys your original code expects
        return {
            "output_text": answer_text,
            "input_documents": docs,
        }

    return chain


def get_answer(query, chain, docembeddings):
    """
    Use the vector store + chain to answer the question.
    This is the same shape your original code used.
    """
    # Retrieve top-k chunks with similarity scores
    relevant_chunks = docembeddings.similarity_search_with_score(query, k=2)
    chunk_docs = [chunk[0] for chunk in relevant_chunks]

    results = chain({"input_documents": chunk_docs, "question": query})

    text_reference = "".join(doc.page_content for doc in results["input_documents"])
    output = {"Answer": results["output_text"], "Reference": text_reference}
    return output


# Load and cache data
docembeddings = load_documents_and_get_embeddings()
chain = get_chain()

query = st.text_input("Enter your question:")
if query:
    with st.spinner("Generating answer..."):
        output = get_answer(query, chain, docembeddings)
        st.subheader("Answer:")
        st.write(output["Answer"])
        st.subheader("Reference:")
        st.write(output["Reference"])
