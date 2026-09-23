from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services.ingestion import IngestionService


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


ALLOWED_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
}


@router.post("/upload")
def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {extension}. "
                f"Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    try:
        with NamedTemporaryFile(
            suffix=extension,
            delete=True,
        ) as temp_file:
            temp_file.write(file.file.read())
            temp_file.flush()

            service = IngestionService(db)

            document = service.ingest(
                temp_file.name
            )

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed.",
        ) from exc

    return {
        "id": document.id,
        "filename": document.filename,
        "title": document.title,
        "message": "Document uploaded and ingested successfully.",
    }