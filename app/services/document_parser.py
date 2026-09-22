from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class ParsedPage:
    page_number: int
    content: str
    section: str | None = None

@dataclass
class ParsedBlock:
    content: str
    section: str | None = None
    page_number: int | None = None


@dataclass
class ParsedDocument:
    filename: str
    file_type: str
    pages: list[ParsedPage]

    @property
    def full_text(self) -> str:
        return "\n\n".join(
            page.content
            for page in self.pages
            if page.content.strip()
        )

class DocumentParser:
    SUPPORTED_EXTENSIONS = {
        ".txt",
        ".md",
        ".pdf",
    }

    def parse(self, file_path: str) -> ParsedDocument:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}"
            )

        if extension == ".pdf":
            return self._parse_pdf(path)

        return self._parse_text_file(
            path,
            extension,
        )

    def _parse_text_file(
        self,
        path: Path,
        extension: str,
    ) -> ParsedDocument:
        content = path.read_text(
            encoding="utf-8",
        )

        return ParsedDocument(
            filename=path.name,
            file_type=extension.lstrip("."),
            pages=[
                ParsedPage(
                    page_number=1,
                    content=content,
                )
            ],
        )

    def _parse_pdf(
        self,
        path: Path,
    ) -> ParsedDocument:
        reader = PdfReader(str(path))

        pages = []

        for index, page in enumerate(reader.pages):
            content = page.extract_text() or ""

            pages.append(
                ParsedPage(
                    page_number=index + 1,
                    content=content,
                )
            )

        return ParsedDocument(
            filename=path.name,
            file_type="pdf",
            pages=pages,
        )