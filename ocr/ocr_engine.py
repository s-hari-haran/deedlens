"""
DeedLens OCR Engine
Extracts text from PDFs and images using Tesseract and EasyOCR.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

from PIL import Image

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import groq
    import base64
    import io
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class OCRBackend(Enum):
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    GROQ = "groq"


@dataclass
class OCRResult:
    """Result from OCR processing."""
    text: str
    confidence: float
    page_number: int
    backend: str


@dataclass
class DocumentOCRResult:
    """Complete OCR result for a document."""
    file_path: str
    pages: List[OCRResult]
    full_text: str
    avg_confidence: float
    total_pages: int


class OCREngine:
    """
    Multi-backend OCR engine supporting Tesseract and EasyOCR.
    Handles PDFs and images.
    """
    
    def __init__(
        self, 
        backend: OCRBackend = OCRBackend.TESSERACT,
        languages: List[str] = None
    ):
        self.backend = backend
        self.languages = languages or ["en"]
        self._easyocr_reader = None
        self._groq_client = None
        
        self._validate_backend()
    
    def _validate_backend(self):
        """Validate that the selected backend is available."""
        if self.backend == OCRBackend.TESSERACT and not TESSERACT_AVAILABLE:
            raise ImportError("Tesseract not available. Install pytesseract.")
        if self.backend == OCRBackend.EASYOCR and not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR not available. Install easyocr.")
        if self.backend == OCRBackend.GROQ and not GROQ_AVAILABLE:
            raise ImportError("Groq client not available. Install groq.")
    
    def _get_easyocr_reader(self):
        """Lazy load EasyOCR reader."""
        if self._easyocr_reader is None:
            self._easyocr_reader = easyocr.Reader(self.languages, gpu=False)
        return self._easyocr_reader
    
    def _pdf_to_images(self, pdf_path: str, dpi: int = 200) -> List[Image.Image]:
        """Convert PDF pages to images."""
        if PDF2IMAGE_AVAILABLE:
            return convert_from_path(pdf_path, dpi=dpi)
        elif PYMUPDF_AVAILABLE:
            doc = fitz.open(pdf_path)
            images = []
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
            doc.close()
            return images
        else:
            raise ImportError("No PDF library available. Install pdf2image or PyMuPDF.")
    
    def _ocr_image_tesseract(self, image: Image.Image) -> Tuple[str, float]:
        """OCR an image using Tesseract."""
        # Get detailed data for confidence
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        # Extract text
        text = pytesseract.image_to_string(image)
        
        # Calculate average confidence (excluding -1 values)
        confidences = [int(c) for c in data['conf'] if int(c) > 0]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        
        return text, avg_conf / 100.0
    
    def _ocr_image_easyocr(self, image: Image.Image) -> Tuple[str, float]:
        """OCR an image using EasyOCR."""
        import numpy as np
        
        reader = self._get_easyocr_reader()
        
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Get OCR results
        results = reader.readtext(img_array)
        
        if not results:
            return "", 0.0
        
        # Extract text and confidence
        texts = []
        confidences = []
        for (bbox, text, conf) in results:
            texts.append(text)
            confidences.append(conf)
        
        full_text = " ".join(texts)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        
        return full_text, avg_conf
    
    def _ocr_image_groq(self, image: Image.Image) -> Tuple[str, float]:
        """OCR an image using Groq Vision."""
        if self._groq_client is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            self._groq_client = groq.Groq(api_key=api_key)
            
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Verified working model
        model = "meta-llama/llama-4-scout-17b-16e-instruct"
        
        try:
            chat_completion = self._groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all text from this property document image exactly as it appears. Do not summarize. Just output the text."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_str}",
                                },
                            },
                        ],
                    }
                ],
                model=model,
                temperature=0.0,
            )
            
            text = chat_completion.choices[0].message.content
            return text, 0.95
            
        except Exception as e:
            print(f"Groq OCR Failed: {str(e)}")
            return "", 0.0


    def _ocr_image(self, image: Image.Image) -> Tuple[str, float]:
        """OCR an image using the configured backend."""
        if self.backend == OCRBackend.TESSERACT:
            return self._ocr_image_tesseract(image)
        elif self.backend == OCRBackend.GROQ:
            return self._ocr_image_groq(image)
        else:
            return self._ocr_image_easyocr(image)
    
    def process_image(self, image_path: str) -> OCRResult:
        """Process a single image file."""
        image = Image.open(image_path)
        text, confidence = self._ocr_image(image)
        
        return OCRResult(
            text=text,
            confidence=confidence,
            page_number=1,
            backend=self.backend.value
        )
    
    def process_pdf(self, pdf_path: str, dpi: int = 200) -> DocumentOCRResult:
        """Process a PDF document."""
        images = self._pdf_to_images(pdf_path, dpi)
        
        pages = []
        all_text = []
        total_conf = 0.0
        
        for i, image in enumerate(images):
            text, confidence = self._ocr_image(image)
            
            pages.append(OCRResult(
                text=text,
                confidence=confidence,
                page_number=i + 1,
                backend=self.backend.value
            ))
            
            all_text.append(f"--- Page {i + 1} ---\n{text}")
            total_conf += confidence
        
        avg_confidence = total_conf / len(pages) if pages else 0.0
        
        return DocumentOCRResult(
            file_path=pdf_path,
            pages=pages,
            full_text="\n\n".join(all_text),
            avg_confidence=avg_confidence,
            total_pages=len(pages)
        )
    
    def process_file(self, file_path: str) -> DocumentOCRResult:
        """Process any supported file (PDF or image)."""
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext == ".pdf":
            return self.process_pdf(file_path)
        elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            result = self.process_image(file_path)
            return DocumentOCRResult(
                file_path=file_path,
                pages=[result],
                full_text=result.text,
                avg_confidence=result.confidence,
                total_pages=1
            )
        else:
            raise ValueError(f"Unsupported file format: {ext}")


def extract_text_from_document(
    file_path: str,
    backend: str = "tesseract",
    languages: List[str] = None
) -> Dict:
    """
    Convenience function to extract text from a document.
    
    Args:
        file_path: Path to PDF or image file
        backend: 'tesseract' or 'easyocr'
        languages: List of language codes
    
    Returns:
        Dictionary with extracted text and metadata
    """
    ocr_backend = OCRBackend(backend)
    engine = OCREngine(backend=ocr_backend, languages=languages)
    result = engine.process_file(file_path)
    
    return {
        "file_path": result.file_path,
        "text": result.full_text,
        "confidence": result.avg_confidence,
        "total_pages": result.total_pages,
        "pages": [
            {
                "page_number": p.page_number,
                "text": p.text,
                "confidence": p.confidence
            }
            for p in result.pages
        ]
    }


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        result = extract_text_from_document(file_path)
        print(f"Extracted {result['total_pages']} pages")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\n--- Text ---")
        print(result['text'][:2000])  # Print first 2000 chars
    else:
        print("Usage: python ocr_engine.py <file_path>")
