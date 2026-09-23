import re
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    content: str
    chunk_index: int
    start_char: int | None = None
    end_char: int | None = None
    section: str | None = None
    page_number: int | None = None


class ChunkingService:
    def __init__(
        self,
        max_chunk_size: int = 1000,
        overlap: int = 150,
    ):
        if max_chunk_size <= 0:
            raise ValueError(
                "max_chunk_size must be greater than 0"
            )

        if overlap < 0:
            raise ValueError(
                "overlap cannot be negative"
            )

        if overlap >= max_chunk_size:
            raise ValueError(
                "overlap must be smaller than max_chunk_size"
            )

        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[Chunk]:
        text = self._normalize_text(text)

        if not text:
            return []

        paragraphs = self._split_paragraphs(text)

        chunks = []

        for paragraph in paragraphs:
            if len(paragraph) <= self.max_chunk_size:
                chunks.append(paragraph)
                continue

            paragraph_chunks = self._split_large_paragraph(
                paragraph
            )

            chunks.extend(paragraph_chunks)

        chunks = self._apply_overlap(chunks)

        return [
            Chunk(
                content=content,
                chunk_index=index,
            )
            for index, content in enumerate(chunks)
        ]

    def _normalize_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    def _split_paragraphs(
        self,
        text: str,
    ) -> List[str]:
        paragraphs = re.split(
            r"\n\s*\n",
            text,
        )

        return [
            paragraph.strip()
            for paragraph in paragraphs
            if paragraph.strip()
        ]

    def _split_large_paragraph(
        self,
        paragraph: str,
    ) -> List[str]:
        sentences = self._split_sentences(paragraph)

        chunks = []
        current = ""

        for sentence in sentences:
            if len(sentence) > self.max_chunk_size:
                if current:
                    chunks.append(current)
                    current = ""

                chunks.extend(
                    self._split_large_sentence(sentence)
                )

                continue

            candidate = self._join_text(
                current,
                sentence,
            )

            if len(candidate) <= self.max_chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)

                current = sentence

        if current:
            chunks.append(current)

        return chunks

    def _split_sentences(
        self,
        text: str,
    ) -> List[str]:
        sentences = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    def _split_large_sentence(
        self,
        sentence: str,
    ) -> List[str]:
        words = sentence.split()

        chunks = []
        current_words = []
        current_length = 0

        for word in words:
            additional_length = len(word)

            if current_words:
                additional_length += 1

            if (
                current_length + additional_length
                <= self.max_chunk_size
            ):
                current_words.append(word)
                current_length += additional_length
            else:
                chunks.append(
                    " ".join(current_words)
                )

                current_words = [word]
                current_length = len(word)

        if current_words:
            chunks.append(
                " ".join(current_words)
            )

        return chunks

    def _apply_overlap(
        self,
        chunks: List[str],
    ) -> List[str]:
        if self.overlap == 0:
            return chunks

        result = []

        for index, chunk in enumerate(chunks):
            if index == 0:
                result.append(chunk)
                continue

            previous_chunk = result[-1]

            overlap_text = self._get_word_overlap(
                previous_chunk
            )

            if not overlap_text:
                result.append(chunk)
                continue

            available_space = (
                self.max_chunk_size
                - len(overlap_text)
                - 1
            )

            if available_space <= 0:
                result.append(chunk)
                continue

            words = chunk.split()
            selected_words = []
            current_length = 0

            for word in words:
                additional_length = len(word)

                if selected_words:
                    additional_length += 1

                if (
                    current_length + additional_length
                    > available_space
                ):
                    break

                selected_words.append(word)
                current_length += additional_length

            if selected_words:
                chunk = (
                    overlap_text
                    + " "
                    + " ".join(selected_words)
                )

            result.append(chunk)

        return result

    def _get_word_overlap(
        self,
        text: str,
    ) -> str:
        words = text.split()

        if not words:
            return ""

        overlap_words = []
        current_length = 0

        for word in reversed(words):
            additional_length = len(word)

            if overlap_words:
                additional_length += 1

            if (
                current_length + additional_length
                > self.overlap
            ):
                break

            overlap_words.insert(0, word)
            current_length += additional_length

        return " ".join(overlap_words)

    def _join_text(
        self,
        current: str,
        new_text: str,
    ) -> str:
        if not current:
            return new_text

        return f"{current} {new_text}"

    def chunk_document(
        self,
        pages: list,
    ) -> List[Chunk]:
        all_chunks = []
        chunk_index = 0

        for page in pages:
            page_chunks = self.chunk_text(
                page.content
            )

            for chunk in page_chunks:
                chunk.chunk_index = chunk_index
                chunk.page_number = page.page_number

                all_chunks.append(chunk)

                chunk_index += 1

        return all_chunks

    def chunk_blocks(
        self,
        blocks: list,
    ) -> List[Chunk]:
        all_chunks = []
        chunk_index = 0

        for block in blocks:
            block_chunks = self.chunk_text(
                block.content
            )

            for chunk in block_chunks:
                chunk.chunk_index = chunk_index
                chunk.page_number = block.page_number
                chunk.section = block.section

                all_chunks.append(chunk)

                chunk_index += 1

        return all_chunks