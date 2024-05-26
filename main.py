

import os
import streamlit as st
from pinecone import Pinecone

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import Settings
import os
from dotenv import load_dotenv


load_dotenv()
st.title("Document Query App")
st.write("This app allows you to query documents for specific information.")
DATA_URL="https://atcold.github.io/NYU-DLSP20/en/week12/12-3/"

llm = Gemini(model_name="models/gemini-1.0-pro",api_key=os.environ["GOOGLE_API_KEY"])

embed_model=GeminiEmbedding(model_name="models/embedding-001")

Settings.llm=llm
Settings.embed_model=embed_model
Settings.chunk_size=1024



# Set the Pinecone API key
api_key = os.environ.get("PINECONE_API_KEY")

# Create an instance of the Pinecone class
pc = Pinecone(api_key=api_key)
pinecone_client=Pinecone(api_key=api_key)
#index_description=pinecone_client.describe_index("demo")
#print(index_description)
loader=BeautifulSoupWebReader()
documents=loader.load_data(urls=[DATA_URL])
st.write("Loaded documents:")
st.write(documents)

pinecone_index=pinecone_client.Index("demo")
vector_store=PineconeVectorStore(pinecone_index=pinecone_index)
pipeline=IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1024,chunk_overlap=20),
        embed_model
    ],
    vector_store=vector_store
)
#pipeline.run(documents=documents)
index=VectorStoreIndex.from_vector_store(vector_store=vector_store)
retriever=VectorIndexRetriever(index=index,similarity_top_k=60)
query_engine=RetrieverQueryEngine(retriever=retriever)
#response=query_engine.query("what is a key-value store")
#print(response)
user_query = st.text_input("Enter your query:")
if user_query:
    response = query_engine.query(user_query)
    st.write("Response:")
    st.write(response)
