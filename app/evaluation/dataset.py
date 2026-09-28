"""
RAGForge Evaluation Dataset
Ground-truth query-to-chunk mappings extracted and verified against
'Building Python Web APIs with FastAPI' (322 chunks indexed).
"""

EVALUATION_DATASET = [
    {
        "id": "q01",
        "question": "What is FastAPI and what are its core features?",
        "relevant_chunk_ids": [44, 45, 51],
    },
    {
        "id": "q02",
        "question": "How do you define a path parameter in FastAPI?",
        "relevant_chunk_ids": [77, 76],
    },
    {
        "id": "q03",
        "question": "How do you run a FastAPI application using Uvicorn?",
        "relevant_chunk_ids": [52, 53],
    },
    {
        "id": "q04",
        "question": "How do you define query parameters in FastAPI route handlers?",
        "relevant_chunk_ids": [78, 56],
    },
    {
        "id": "q05",
        "question": "What is dependency injection in FastAPI?",
        "relevant_chunk_ids": [217, 218, 219],
    },
    {
        "id": "q06",
        "question": "How does FastAPI use Pydantic models?",
        "relevant_chunk_ids": [67, 91, 136],
    },
    {
        "id": "q07",
        "question": "How do you structure larger FastAPI projects using APIRouter?",
        "relevant_chunk_ids": [55, 56, 131, 132],
    },
    {
        "id": "q08",
        "question": "How do you configure CORS middleware in FastAPI applications?",
        "relevant_chunk_ids": [258, 259, 260],
    },
    {
        "id": "q09",
        "question": "How can response models be defined in FastAPI?",
        "relevant_chunk_ids": [92, 93, 94],
    },
    {
        "id": "q10",
        "question": "What is HTTPException used for in FastAPI?",
        "relevant_chunk_ids": [100, 101],
    },
    {
        "id": "q11",
        "question": "How do you handle authentication in FastAPI?",
        "relevant_chunk_ids": [215, 212, 236, 240],
    },
    {
        "id": "q12",
        "question": "How is password hashing implemented in FastAPI using passlib CryptContext?",
        "relevant_chunk_ids": [224, 223],
    },
    {
        "id": "q13",
        "question": "How do you create access tokens with expiration for user authentication?",
        "relevant_chunk_ids": [231, 233, 238],
    },
    {
        "id": "q14",
        "question": "How do you handle OAuth2 password request forms in sign-in routes?",
        "relevant_chunk_ids": [239, 240],
    },
    {
        "id": "q15",
        "question": "How do you configure Jinja2 templates in FastAPI?",
        "relevant_chunk_ids": [107, 117, 118, 120],
    },
    {
        "id": "q16",
        "question": "How do you set custom HTTP status codes on FastAPI routes?",
        "relevant_chunk_ids": [103, 144, 145],
    },
    {
        "id": "q17",
        "question": "What is SQLModel and how does it connect FastAPI to SQL databases?",
        "relevant_chunk_ids": [169, 170, 171],
    },
    {
        "id": "q18",
        "question": "How does FastAPI generate API documentation?",
        "relevant_chunk_ids": [80, 85],
    },
    {
        "id": "q19",
        "question": "How do you write unit tests for FastAPI using pytest?",
        "relevant_chunk_ids": [264, 267],
    },
    {
        "id": "q20",
        "question": "How do you test FastAPI applications?",
        "relevant_chunk_ids": [264, 263, 299],
    },
]