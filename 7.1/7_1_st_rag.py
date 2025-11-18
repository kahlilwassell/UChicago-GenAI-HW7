import os

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.question_answering import load_qa_chain

# Load environment variables (OPENAI_API_KEY)
load_dotenv()

st.title("RAG with Streamlit (HW 7-1)")


@st.cache_resource
def load_vectorstore():
    """
    Load documents from docs/ and build a Chroma vector store using OpenAI embeddings.
    Uses:
      - PDFs
      - .txt files
      - .md (Markdown) files
    """

    # Load documents of multiple types
    pdf_loader = DirectoryLoader(
        "docs",
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )

    txt_loader = DirectoryLoader(
        "docs",
        glob="*.txt",
        loader_cls=TextLoader,
    )

    md_loader = DirectoryLoader(
        "docs",
        glob="*.md",
        loader_cls=TextLoader,
    )

    documents = []
    for loader in [pdf_loader, txt_loader, md_loader]:
        try:
            documents.extend(loader.load())
        except Exception as e:
            # Don't crash if a loader can't read something; just show a warning
            st.warning(f"Problem loading documents with {loader.__class__.__name__}: {e}")

    if not documents:
        st.error(
            "No documents found in the 'docs' folder. "
            "Add some .pdf, .txt, or .md files and rerun the app."
        )
        return None

    # Chunk documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)

    # Create Chroma vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory="chroma_db",
    )
    return vectorstore


def get_chain():
    """
    Build a simple QA chain over retrieved chunks.
    """

    prompt_template = """
Use the following pieces of context to answer the question at the end.
Make sure the answer *explicitly* leverages the context rather than just general knowledge.
If the context doesn't really answer the question, say that clearly.

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

    # Simple "stuff" chain is enough for this homework
    chain = load_qa_chain(
        OpenAI(temperature=0),
        chain_type="stuff",
        prompt=prompt,
    )
    return chain


def get_answer(query, chain, vectorstore):
    # Retrieve top-k similar chunks
    docs = vectorstore.similarity_search(query, k=3)

    # Run chain
    result = chain({"input_documents": docs, "question": query})

    # Concatenate the reference text so you can see what the model saw
    reference_text = "\n\n---\n\n".join(d.page_content for d in docs)

    return {
        "answer": result["output_text"],
        "reference": reference_text,
    }


# Main app logic

vectorstore = load_vectorstore()
chain = get_chain() if vectorstore is not None else None

query = st.text_input("Enter your question about the documents:")

if query and vectorstore is not None and chain is not None:
    with st.spinner("Retrieving and generating answer..."):
        output = get_answer(query, chain, vectorstore)

    st.subheader("Answer:")
    st.write(output["answer"])

    st.subheader("Reference Chunks (what the model saw):")
    st.write(output["reference"])
elif vectorstore is None:
    st.info(
        "Please add documents into a folder named 'docs' next to this script "
        "and then rerun the app."
    )
