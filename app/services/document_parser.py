from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class ParsedPage:
    page_number: int
    content: str


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
    blocks: list[ParsedBlock]

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

        page = ParsedPage(
            page_number=1,
            content=content,
        )

        if extension == ".md":
            blocks = self._parse_markdown_blocks(
                content,
                page_number=1,
            )
        else:
            blocks = [
                ParsedBlock(
                    content=content.strip(),
                    page_number=1,
                )
            ]

        return ParsedDocument(
            filename=path.name,
            file_type=extension.lstrip("."),
            pages=[page],
            blocks=blocks,
        )

    def _parse_pdf(
        self,
        path: Path,
    ) -> ParsedDocument:
        reader = PdfReader(str(path))

        pages = []
        blocks = []

        for index, page in enumerate(reader.pages):
            page_number = index + 1
            content = page.extract_text() or ""

            pages.append(
                ParsedPage(
                    page_number=page_number,
                    content=content,
                )
            )

            if content.strip():
                blocks.append(
                    ParsedBlock(
                        content=content.strip(),
                        page_number=page_number,
                    )
                )

        return ParsedDocument(
            filename=path.name,
            file_type="pdf",
            pages=pages,
            blocks=blocks,
        )

    def _parse_markdown_blocks(
        self,
        content: str,
        page_number: int,
    ) -> list[ParsedBlock]:
        lines = content.splitlines()

        blocks = []
        current_section = None
        current_lines = []

        def flush_block():
            if not current_lines:
                return

            block_content = "\n".join(
                current_lines
            ).strip()

            if block_content:
                blocks.append(
                    ParsedBlock(
                        content=block_content,
                        section=current_section,
                        page_number=page_number,
                    )
                )

            current_lines.clear()

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("#"):
                heading = stripped.lstrip("#").strip()

                if heading:
                    flush_block()
                    current_section = heading

                continue

            current_lines.append(line)

        flush_block()

        return blocks