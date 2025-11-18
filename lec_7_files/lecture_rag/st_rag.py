import streamlit as st
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.output_parsers import RegexParser
from langchain_openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

st.title("RAG with Streamlit")

@st.cache_resource
def load_documents_and_get_embeddings():
    loader = DirectoryLoader('docs', glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    chunk_size = 1000
    chunk_overlap = 100
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    index_path = "llm_faiss_index"
    if os.path.exists(index_path):
        docembeddings = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        docembeddings = FAISS.from_documents(texts, embeddings)
        docembeddings.save_local(index_path)
    return docembeddings

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
Helpful Answer:"""

    output_parser = RegexParser(
        regex=r"(.*?)\nScore: (.*)",
        output_keys=["answer", "score"],
    )

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
        output_parser=output_parser
    )

    chain = load_qa_chain(
        OpenAI(temperature=0),
        chain_type="map_rerank",
        return_intermediate_steps=True,
        prompt=PROMPT
    )
    return chain

def get_answer(query, chain, docembeddings):
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