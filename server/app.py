import os
from dotenv import load_dotenv
import tempfile
import PyPDF2
from pinecone import Pinecone
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables (set these in your environment)
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_GENAI_KEY = os.getenv("GOOGLE_GENAI_KEY")
PINECONE_INDEX_NAME = "chatpdf"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Matching model dimensions

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust for your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Google AI
genai.configure(api_key=GOOGLE_GENAI_KEY)

# Load Sentence Transformer Model
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure Index Exists
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec={"serverless": {"cloud": "aws", "region": "us-west-2"}},
    )

# Connect to the index
index = pc.Index(PINECONE_INDEX_NAME)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_bytes, chunk_size=500):
    with tempfile.NamedTemporaryFile(delete=True) as temp_pdf:
        temp_pdf.write(pdf_bytes)
        temp_pdf.flush()
        
        with open(temp_pdf.name, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = " ".join([page.extract_text() or "" for page in reader.pages])
    
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to store text chunks in Pinecone
def store_text_in_pinecone(pdf_bytes):
    chunks = extract_text_from_pdf(pdf_bytes)
    vectors = [
        {
            "id": f"chunk_{i}",
            "values": embedding_model.encode(chunk).tolist(),
            "metadata": {"text": chunk},
        }
        for i, chunk in enumerate(chunks)
    ]
    index.upsert(vectors)

# Function to retrieve relevant chunks
def retrieve_relevant_chunks(query, top_k=3):
    query_vector = embedding_model.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    return " ".join(match.metadata.get("text", "") for match in results.get("matches", []))

# AI Query Function
def ask_study_ai(query):
    retrieved_text = retrieve_relevant_chunks(query)
    if not retrieved_text:
        return "I couldn't find relevant information in the study material."
    
    prompt = f"Use the following study material to answer:\n{retrieved_text}\n\nQuestion: {query}"
    response = genai.GenerativeModel("gemini-1.5-pro").generate_content(prompt)
    return response.text

# Request Model
class QueryRequest(BaseModel):
    query: str

# API Endpoints
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    store_text_in_pinecone(pdf_bytes)
    return {"message": "PDF processed and stored in vector database successfully."}

@app.post("/ask")
async def ask_question(request: QueryRequest):
    answer = ask_study_ai(request.query)
    return {"answer": answer}