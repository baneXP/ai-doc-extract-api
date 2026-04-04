# core python libraries
import os       # read environment variables
import base64   # decode incoming base64 file string back to bytes
import time     # measure processing time per request

# fastapi — web framework that creates API endpoints
from fastapi import FastAPI, HTTPException, Header
# FastAPI -> main app class
# HTTPException -> return error responses with status codes
# Header -> read values from request headers (like x-api-key)

from pydantic import BaseModel  # validates request/response structure automatically
from dotenv import load_dotenv  # loads .env file into environment variables

# .env is one level up from src/ — build the path dynamically
# __file__ = this file's path, dirname = its folder, .. = one level up
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# import our own modules — order matters, .env must load first
from extractor import extract_text        # handles PDF/DOCX/image text extraction
from nlp_pipeline import analyze_document  # handles Groq LLM analysis


# create the FastAPI app — this is the main entry point
app = FastAPI(
    title="AI-Powered Document Analysis API",
    description="Extracts summary, entities, and sentiment from PDF, DOCX, and image documents.",
    version="1.0.0"
    # visiting /docs auto-generates Swagger UI from this metadata
)

# load our custom API key from .env — used to authenticate incoming requests
API_KEY = os.getenv("API_KEY")


# Pydantic models define the shape of JSON going in and out
# if request doesn't match → FastAPI auto-returns 422 error

class DocumentRequest(BaseModel):
    # what GUVI sends us
    fileName: str    # original file name e.g. "invoice.pdf"
    fileType: str    # pdf / docx / image
    fileBase64: str  # the file contents encoded as base64 string


class EntitiesResponse(BaseModel):
    # named entities extracted from document
    names: list[str]          # person names
    dates: list[str]          # dates found
    organizations: list[str]  # company/org names
    amounts: list[str]        # monetary values


class AnalysisResponse(BaseModel):
    # what we send back to GUVI — must match their expected schema exactly
    status: str              # "success" or "error"
    fileName: str            # echo back the file name
    summary: str             # AI-generated summary
    entities: EntitiesResponse
    sentiment: str           # "Positive" / "Neutral" / "Negative"


class ErrorResponse(BaseModel):
    status: str
    message: str


# Routes connect a URL path + HTTP method to a Python function

@app.get("/")
def root():
    # health check — confirms API is alive
    return {
        "message": "Document Analysis API is running",
        "endpoint": "POST /api/document-analyze",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    # simple ping endpoint — useful for monitoring
    return {"status": "ok"}


@app.post("/api/document-analyze", response_model=AnalysisResponse)
def analyze(request: DocumentRequest, x_api_key: str = Header(...)):
    # Header(...) -> FastAPI reads x-api-key from request headers automatically
    # ... means required — missing header = 422 before our code even runs

    #Auth
    # reject requests with wrong or missing API key
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,  # 401 = Unauthorized
            detail="Unauthorized: Invalid or missing API key"
        )

    start = time.time()  # start timer to measure total processing time

    #ValidatefileType
    # only accept formats we can actually process
    supported = ["pdf", "docx", "image"]
    if request.fileType.lower() not in supported:
        raise HTTPException(
            status_code=415,  # 415 = Unsupported Media Type
            detail=f"Unsupported fileType '{request.fileType}'. Must be one of: {supported}"
        )

    #DecodeBase64
    try:
        file_bytes = base64.b64decode(request.fileBase64)
    except Exception:
        raise HTTPException(
            status_code=422,  # 422 = Unprocessable Entity
            detail="Invalid base64 encoding in fileBase64 field"
        )

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=422,
            detail="Decoded file is empty"
        )

    #Extract Text
    # delegate to extractor.py — it picks the right extractor based on fileType
    try:
        extracted_text = extract_text(file_bytes, request.fileType)
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,  # 500 = Internal Server Error
            detail=f"Text extraction failed: {str(e)}"
        )

    #NLP Analysis
    # delegate to nlp_pipeline.py — sends text to Groq LLM, gets structured JSON back
    from nlp_pipeline import analyze_image_directly

    try:
        # If OCR failed or weak → use vision model
        if len(extracted_text.strip()) < 30 and request.fileType.lower() == "image":
            print("[INFO] Using vision model instead of OCR...")
            analysis = analyze_image_directly(file_bytes)
        else:
            analysis = analyze_document(extracted_text)

    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Analysis service unavailable: {str(e)}"
        )

    elapsed = round(time.time() - start, 2)
    print(f"[INFO] Processed '{request.fileName}' in {elapsed}s")

    #Build Response
    # **analysis["entities"] unpacks dict keys as keyword arguments into EntitiesResponse
    return AnalysisResponse(
        status="success",
        fileName=request.fileName,
        summary=analysis["summary"],
        entities=EntitiesResponse(**analysis["entities"]),
        sentiment=analysis["sentiment"]
    )
