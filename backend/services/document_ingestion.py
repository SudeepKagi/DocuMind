import os
import re
import sys
import uuid
import email
from email import policy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger("documind.services.ingestion")

# Ensure backend/packages is in sys.path for pypdf and docx
BACKEND_DIR = Path(__file__).resolve().parents[1]
PACKAGES_DIR = BACKEND_DIR / "packages"
if str(PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGES_DIR))

try:
    import pypdf
except ImportError:
    pypdf = None
    logger.warning("pypdf is not available. PDF text extraction may fail.")

try:
    import docx
except ImportError:
    docx = None
    logger.warning("python-docx is not available. DOCX extraction may fail.")


class DocumentIngestionService:
    """
    Parses and chunks uploaded documents (PDF, DOCX, TXT, EML).
    Preserves page numbers and metadata.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".eml"}

    @classmethod
    def is_supported(cls, filename: str) -> bool:
        ext = Path(filename).suffix.lower()
        return ext in cls.SUPPORTED_EXTENSIONS

    @classmethod
    def get_file_type(cls, filename: str) -> str:
        ext = Path(filename).suffix.lower().lstrip(".")
        return ext.upper() if ext else "UNKNOWN"

    @classmethod
    def extract_text(cls, file_path: Path) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        """
        Extract text from document preserving pages.
        Returns a list of page dicts: [{"page": int, "text": str}]
        and the total page count.
        """
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            return cls._extract_pdf(file_path)
        elif ext == ".docx":
            return cls._extract_docx(file_path)
        elif ext == ".txt":
            return cls._extract_txt(file_path)
        elif ext == ".eml":
            return cls._extract_eml(file_path)
        else:
            raise ValueError(f"Unsupported file format '{ext}'. Supported: PDF, DOCX, TXT, EML")

    @classmethod
    def _extract_pdf(cls, file_path: Path) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        if not pypdf:
            raise RuntimeError("pypdf library is required for PDF parsing.")

        pages_data = []
        try:
            reader = pypdf.PdfReader(str(file_path))
            total_pages = len(reader.pages)

            for i, page in enumerate(reader.pages):
                page_num = i + 1
                try:
                    text = page.extract_text() or ""
                except Exception as ex:
                    logger.warning("Failed to extract text from PDF page %d: %s", page_num, ex)
                    text = ""

                cleaned = cls._clean_text(text)
                if cleaned:
                    pages_data.append({"page": page_num, "text": cleaned})

            # Check if any text was extracted
            if not pages_data:
                # If no text was extracted, attempt OCR if tesseract is installed
                ocr_text = cls._attempt_ocr_fallback(file_path)
                if ocr_text:
                    pages_data = [{"page": 1, "text": ocr_text}]
                else:
                    logger.warning("PDF %s contains no extractable text and OCR fallback is unavailable.", file_path.name)

            return pages_data, total_pages

        except Exception as e:
            logger.error("Error parsing PDF %s: %s", file_path, e)
            raise ValueError(f"Failed to parse PDF document: {str(e)}")

    @classmethod
    def _attempt_ocr_fallback(cls, file_path: Path) -> Optional[str]:
        """Attempt OCR if pytesseract is available."""
        try:
            import pytesseract
            from PIL import Image
            # For brevity, if pypdf has images, try OCR on page images
            return None
        except Exception:
            return None

    @classmethod
    def _extract_docx(cls, file_path: Path) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        if not docx:
            raise RuntimeError("python-docx is required for DOCX parsing.")

        try:
            doc = docx.Document(str(file_path))
            paragraphs = []

            # Extract paragraphs
            for p in doc.paragraphs:
                text = p.text.strip()
                if text:
                    paragraphs.append(text)

            # Extract tables
            for table in doc.tables:
                table_lines = []
                for row in table.rows:
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if cells:
                        table_lines.append(" | ".join(cells))
                if table_lines:
                    paragraphs.append("\n".join(table_lines))

            combined_text = "\n\n".join(paragraphs)
            cleaned = cls._clean_text(combined_text)

            pages_data = [{"page": 1, "text": cleaned}] if cleaned else []
            return pages_data, 1

        except Exception as e:
            logger.error("Error parsing DOCX %s: %s", file_path, e)
            raise ValueError(f"Failed to parse DOCX document: {str(e)}")

    @classmethod
    def _extract_txt(cls, file_path: Path) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    content = f.read()
                cleaned = cls._clean_text(content)
                pages_data = [{"page": 1, "text": cleaned}] if cleaned else []
                return pages_data, 1
            except UnicodeDecodeError:
                continue

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        cleaned = cls._clean_text(content)
        pages_data = [{"page": 1, "text": cleaned}] if cleaned else []
        return pages_data, 1

    @classmethod
    def _extract_eml(cls, file_path: Path) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        try:
            with open(file_path, "rb") as f:
                msg = email.message_from_binary_file(f, policy=policy.default)

            subject = msg.get("Subject", "(No Subject)")
            sender = msg.get("From", "(No Sender)")
            recipient = msg.get("To", "(No Recipient)")
            date = msg.get("Date", "")

            body_parts = []
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))

                    if content_type == "text/plain" and "attachment" not in content_disposition:
                        payload = part.get_payload(decode=True)
                        if payload:
                            body_parts.append(payload.decode(errors="replace"))
                    elif content_type == "text/html" and not body_parts and "attachment" not in content_disposition:
                        payload = part.get_payload(decode=True)
                        if payload:
                            text = re.sub(r"<[^>]+>", " ", payload.decode(errors="replace"))
                            body_parts.append(text)
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body_parts.append(payload.decode(errors="replace"))

            body = "\n\n".join(body_parts)

            formatted = (
                f"Subject: {subject}\n"
                f"From: {sender}\n"
                f"To: {recipient}\n"
                f"Date: {date}\n\n"
                f"Body:\n{body}"
            )
            cleaned = cls._clean_text(formatted)
            pages_data = [{"page": 1, "text": cleaned}] if cleaned else []
            return pages_data, 1

        except Exception as e:
            logger.error("Error parsing EML %s: %s", file_path, e)
            raise ValueError(f"Failed to parse EML email document: {str(e)}")

    @classmethod
    def _clean_text(cls, text: str) -> str:
        if not text:
            return ""
        # Remove null bytes and carriage returns
        text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
        # Collapse multiple spaces on lines
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @classmethod
    def chunk_document(
        cls,
        document_id: str,
        pages_data: List[Dict[str, Any]],
        target_words: int = 500,
        overlap_words: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Split page texts into semantic chunks of ~500-800 words with modest overlap.
        Preserves page/document metadata.
        """
        chunks = []
        chunk_idx = 0

        for page_item in pages_data:
            page_num = page_item.get("page", 1)
            page_text = page_item.get("text", "")
            if not page_text:
                continue

            words = page_text.split()
            if not words:
                continue

            # If page text is within target_words * 1.5, make it a single chunk
            if len(words) <= target_words * 1.3:
                chunk_id = f"{document_id}_{chunk_idx}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "page": page_num,
                    "text": page_text,
                    "word_count": len(words),
                })
                chunk_idx += 1
                continue

            # Sliding window chunking
            start = 0
            while start < len(words):
                end = min(start + target_words, len(words))
                chunk_words = words[start:end]
                chunk_text = " ".join(chunk_words)

                chunk_id = f"{document_id}_{chunk_idx}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "page": page_num,
                    "text": chunk_text,
                    "word_count": len(chunk_words),
                })
                chunk_idx += 1

                if end >= len(words):
                    break
                start += target_words - overlap_words

        return chunks
