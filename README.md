# ComprehensiveQA API

A multi-round retrieval API implementing the Retrieve-Verify-Retrieve (RVR) framework for comprehensive question answering with maximum answer coverage.

## Based on

This project implements the approach described in:

**RVR: Retrieve-Verify-Retrieve for Comprehensive Question Answering**  
Deniz Qian, Hung-Ting Chen, Eunsol Choi  
arXiv: https://arxiv.org/abs/2602.18425v1

## Problem

Traditional single-pass retrieval systems fail to comprehensively answer queries that have multiple valid answers (e.g., "What are treatment options for condition X?", "List legal precedents for Y"). RVR solves this by iteratively retrieving, verifying quality, and re-querying with context to maximize answer coverage.

## Features

- **Multi-round Retrieval**: Iteratively searches for diverse answers
- **Verification Layer**: Filters high-quality, relevant results
- **Query Augmentation**: Enhances queries with verified context from previous rounds
- **REST API**: Easy integration with existing platforms
- **Configurable**: Adjust rounds, retrieval parameters, and verification thresholds

## Use Cases

- Medical Q&A platforms (treatment options, diagnostic criteria)
- Legal research (precedents, statutes, case law)
- Academic research (literature review, methodology comparison)
- Product comparison (features, alternatives)

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

The API will start on `http://localhost:8000`

## Usage

### Query the API

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are effective treatments for type 2 diabetes?",
    "max_rounds": 3,
    "docs_per_round": 5
  }'
```

### Run Demo

```bash
python demo.py
```

The demo shows RVR in action on sample medical and legal queries.

## API Endpoints

### POST /query

Performs multi-round RVR retrieval.

**Request Body:**
```json
{
  "query": "string",
  "max_rounds": 3,
  "docs_per_round": 5,
  "verification_threshold": 0.6
}
```

**Response:**
```json
{
  "query": "string",
  "total_rounds": 3,
  "verified_documents": [
    {
      "text": "document content",
      "score": 0.85,
      "round": 1,
      "source": "document_id"
    }
  ],
  "coverage_improvement": "15.2%"
}
```

## Architecture

1. **Retriever**: Dense retrieval using sentence transformers
2. **Verifier**: Relevance scoring model to filter quality results
3. **Query Augmenter**: Incorporates verified answers into next round queries
4. **Controller**: Orchestrates multi-round iteration

## Configuration

Edit `config.py` to customize:
- Number of retrieval rounds
- Documents per round
- Verification threshold
- Embedding model
- Document corpus

## License

MIT