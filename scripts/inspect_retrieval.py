from app.db.database import SessionLocal
from app.evaluation.evaluator import RetrievalEvaluator


QUESTIONS = [
    "What is FastAPI?",
    "How do you define a path parameter in FastAPI?",
    "What is dependency injection in FastAPI?",
    "How does FastAPI use Pydantic models?",
    "How can response models be defined in FastAPI?",
    "What is HTTPException used for in FastAPI?",
    "How do you handle authentication in FastAPI?",
    "How do you upload files using FastAPI?",
    "How does FastAPI generate API documentation?",
    "How do you test FastAPI applications?",
]


def main():
    db = SessionLocal()

    try:
        evaluator = RetrievalEvaluator(db)

        for question in QUESTIONS:
            evaluator.inspect_question(
                question=question,
                top_k=5,
            )

    finally:
        db.close()


if __name__ == "__main__":
    main()