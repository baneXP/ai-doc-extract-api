# AI-Powered Document Analysis & Extraction API

## Description

This project is an intelligent document processing API that extracts, analyzes, and summarizes content from multiple document formats including PDF, DOCX, and images.

The system leverages OCR and AI models to understand document content, generate summaries, extract key entities (names, dates, organizations, monetary values), and perform sentiment analysis.

---

## Tech Stack

### Language / Framework

* Python
* FastAPI

### Key Libraries

* pytesseract (OCR for images)
* Pillow (image preprocessing)
* python-docx (DOCX parsing)
* PyPDF2 / pdfplumber (PDF parsing)
* python-dotenv (environment variables)

### AI / LLM

* Groq API
* LLaMA 3.3 70B (text analysis)
* LLaMA Vision model (image understanding)

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-doc-extract-api.git
cd ai-doc-extract-api
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set environment variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
API_KEY=your_secret_api_key
```

### 5. Run the application

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

---

## API Authentication

All requests require an API key in header:

```
x-api-key: YOUR_SECRET_API_KEY
```

Requests without a valid API key are rejected.

---

## API Endpoint

```
POST /api/document-analyze
```

---

## Request Format

```json
{
  "fileName": "sample.pdf",
  "fileType": "pdf",
  "fileBase64": "BASE64_ENCODED_FILE"
}
```

---

## Sample Response

```json
{
  "status": "success",
  "fileName": "sample.pdf",
  "summary": "This document describes ...",
  "entities": {
    "names": ["Ravi Kumar"],
    "dates": ["10 March 2026"],
    "organizations": ["ABC Pvt Ltd"],
    "amounts": ["₹10,000"]
  },
  "sentiment": "Neutral"
}
```

---

## Approach

### 1. Document Processing

* PDF → Extracted using PDF parsing libraries
* DOCX → Parsed using python-docx
* Image → Processed using Tesseract OCR

### 2. Text Preprocessing

* Removes noise and invalid characters
* Truncates long documents to fit model limits

### 3. AI Analysis

* Uses LLM to generate:

  * Summary
  * Named entity extraction
  * Sentiment classification

### 4. Post-processing

* Corrects entity misclassification (e.g., organizations vs names)
* Ensures strict JSON output format

### 5. Error Handling

* Retry logic for API failures
* Safe JSON parsing
* Fallback for low-text documents

---

## Features

* Multi-format support (PDF, DOCX, Image)
* OCR-based text extraction
* AI-powered summarization
* Named entity recognition
* Sentiment analysis
* Secure API with key authentication
* Robust error handling

---

## Project Structure

```
ai-doc-extract-api/
├── README.md
├── requirements.txt
├── .env.example
├── src/
│   ├── main.py
│   ├── extractor.py
│   ├── nlp_pipeline.py
```

---

## Deployment

The API is deployed on AWS EC2 and is publicly accessible.

---

## Notes

* Tesseract must be installed on the system for OCR functionality
* Large documents are truncated to ensure performance and avoid model limits

---

## Conclusion

This API provides a scalable and efficient solution for automated document understanding using modern AI techniques. It is suitable for real-world applications such as invoice processing, document classification, and information extraction.
