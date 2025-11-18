import os

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

st.title("Homework 7.1 – RAG with Streamlit (PDF, TXT, MD)")


# Load documents + build or load FAISS index
@st.cache_resource
def load_documents_and_get_vectorstore():
    docs_dir = "./docs"
    if not os.path.isdir(docs_dir):
        return None

    # Load PDFs, text, and markdown files
    loaders = [
        DirectoryLoader(docs_dir, glob="*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(docs_dir, glob="*.txt", loader_cls=TextLoader),
        DirectoryLoader(docs_dir, glob="*.md", loader_cls=TextLoader),
    ]

    documents = []
    for loader in loaders:
        try:
            documents.extend(loader.load())
        except Exception as e:
            print(f"Loader failed: {e}")

    if not documents:
        return None

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    index_dir = "faiss_index_7_1"
    index_file = os.path.join(index_dir, "index.faiss")

    # Load existing index if present
    if os.path.isdir(index_dir) and os.path.exists(index_file):
        vectorstore = FAISS.load_local(
            index_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(index_dir)

    return vectorstore


# Build simple QA "chain"
def get_chain():
    prompt_template = """
Use the following pieces of context to answer the question at the end.  Make sure the answer leverages the context rather than
just your general knowledge. Before answering, ask yourself if you made meaningful use of the  context. If not, just say
it wasn't really covered in the provided material. Triple check that your answer makes explicit use of material included in the context.

This should be in the following format:

Question: [question here]
Helpful Answer: [answer here]
Score: [score between 0 and 100]

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
        docs = inputs["input_documents"]
        question = inputs["question"]

        context = "\n\n".join(d.page_content for d in docs)
        final_prompt = prompt.format(context=context, question=question)

        result = llm.invoke(final_prompt)
        answer = getattr(result, "content", str(result))

        return {
            "output_text": answer,
            "input_documents": docs,
        }

    return chain


# Retrieve + answer
def get_answer(query: str, chain, vectorstore: FAISS):
    retrieved_docs = vectorstore.similarity_search(query, k=4)

    results = chain({"input_documents": retrieved_docs, "question": query})

    reference = "\n\n---\n\n".join(
        doc.page_content for doc in results["input_documents"]
    )

    return {
        "answer": results["output_text"],
        "reference": reference,
    }


# Streamlit UI
vectorstore = load_documents_and_get_vectorstore()
chain = get_chain()

query = st.text_input("Enter your question:")

if query and vectorstore is not None:
    with st.spinner("Retrieving relevant chunks and generating answer..."):
        output = get_answer(query, chain, vectorstore)

    st.subheader("Answer:")
    st.write(output["answer"])

    st.subheader("Reference Chunks Used:")
    st.write(output["reference"])

elif vectorstore is None:
    st.info(
        "Please create a folder named 'docs' and add .pdf, .txt, or .md files. "
        "Then reload this app."
    )
