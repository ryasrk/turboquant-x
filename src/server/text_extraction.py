"""
Text extraction module for TurboQuant-X attachment processing.

Extracts readable text from various document formats including PDF, DOCX, CSV, JSON, YAML,
and plain text files with proper safety measures and rate limiting.
"""

import csv
import io
import json
import logging
from pathlib import Path
from typing import Optional

# Constants for safety limits
MAX_EXTRACTED_LENGTH: int = 50_000  # Maximum characters to extract
MAX_PDF_PAGES: int = 50            # Maximum PDF pages to process
MAX_CSV_ROWS: int = 500            # Maximum CSV rows to process

logger = logging.getLogger(__name__)

# Optional dependency handling
try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    try:
        import pypdf as PyPDF2  # Alternative package name
        HAS_PYPDF2 = True
    except ImportError:
        HAS_PYPDF2 = False
        logger.warning("PyPDF2/pypdf not available. PDF extraction will be limited.")

try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    logger.warning("python-docx not available. DOCX extraction will be limited.")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    logger.warning("PyYAML not available. YAML files will be read as plain text.")

try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False
    logger.warning("chardet not available. Using UTF-8 with error replacement for text files.")


def _truncate_if_needed(text: str) -> str:
    """Truncate text if it exceeds maximum length with appropriate notice."""
    if len(text) > MAX_EXTRACTED_LENGTH:
        truncated = text[:MAX_EXTRACTED_LENGTH]
        truncated += f"\n\n[...truncated at {MAX_EXTRACTED_LENGTH:,} chars]"
        return truncated
    return text


def _detect_encoding(file_path: str) -> str:
    """Detect file encoding using chardet or fall back to UTF-8."""
    if not HAS_CHARDET:
        return 'utf-8'
    
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10240)  # Read first 10KB for detection
        result = chardet.detect(raw_data)
        encoding = result.get('encoding', 'utf-8')
        confidence = result.get('confidence', 0)
        
        if confidence < 0.7:  # Low confidence, fall back to UTF-8
            logger.warning(f"Low encoding confidence ({confidence:.2f}) for {file_path}, using UTF-8")
            return 'utf-8'
        
        logger.debug(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
        return encoding
    except Exception as e:
        logger.warning(f"Encoding detection failed for {file_path}: {e}. Using UTF-8.")
        return 'utf-8'


def extract_text(file_path: str, mime_type: str) -> str:
    """
    Extract readable text from a file based on its MIME type.
    
    Args:
        file_path: Path to the file to extract text from
        mime_type: MIME type of the file
        
    Returns:
        Extracted text content, truncated if necessary
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        PermissionError: If the file can't be read
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Extracting text from {file_path} (MIME type: {mime_type})")
    
    try:
        # Route to appropriate extractor based on MIME type
        if mime_type == 'application/pdf':
            text = extract_pdf(file_path)
        elif mime_type in ('text/plain', 'text/markdown', 'text/x-python'):
            text = extract_plaintext(file_path)
        elif mime_type == 'application/json':
            text = extract_json(file_path)
        elif mime_type == 'text/csv':
            text = extract_csv(file_path)
        elif mime_type in ('application/x-yaml', 'text/yaml'):
            text = extract_yaml(file_path)
        elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            text = extract_docx(file_path)
        else:
            # Fall back to plain text extraction for unknown types
            logger.warning(f"Unknown MIME type {mime_type}, attempting plain text extraction")
            text = extract_plaintext(file_path)
        
        return _truncate_if_needed(text)
        
    except Exception as e:
        error_msg = f"Failed to extract text from {file_path}: {str(e)}"
        logger.error(error_msg)
        return f"Error extracting text: {error_msg}"


def extract_pdf(file_path: str) -> str:
    """
    Extract text from PDF file using PyPDF2.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text from all pages (up to MAX_PDF_PAGES)
    """
    if not HAS_PYPDF2:
        return "PDF extraction requires PyPDF2. Install with: pip install PyPDF2"
    
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            # Limit number of pages processed
            num_pages = min(len(reader.pages), MAX_PDF_PAGES)
            if len(reader.pages) > MAX_PDF_PAGES:
                logger.warning(f"PDF has {len(reader.pages)} pages, processing only first {MAX_PDF_PAGES}")
            
            text_content = []
            for page_num in range(num_pages):
                try:
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    if text.strip():  # Only add non-empty pages
                        text_content.append(f"--- Page {page_num + 1} ---\n{text}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                    continue
            
            extracted_text = '\n\n'.join(text_content)
            
            if not extracted_text.strip():
                logger.warning(f"No text extracted from {file_path} - may be a scanned PDF")
                return "No text content found in PDF. This may be a scanned document requiring OCR."
            
            return extracted_text
            
    except Exception as e:
        logger.error(f"PDF extraction failed for {file_path}: {e}")
        return f"Error reading PDF: {str(e)}"


def extract_plaintext(file_path: str) -> str:
    """
    Read plain text file with automatic encoding detection.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        File contents as string
    """
    encoding = _detect_encoding(file_path)
    
    try:
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            return f.read()
    except UnicodeDecodeError:
        # Fallback to UTF-8 with error replacement
        logger.warning(f"Encoding {encoding} failed, falling back to UTF-8 with error replacement")
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read text file {file_path}: {e}")
        return f"Error reading text file: {str(e)}"


def extract_json(file_path: str) -> str:
    """
    Parse and pretty-print JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Formatted JSON content as string
    """
    try:
        encoding = _detect_encoding(file_path)
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            data = json.load(f)
        
        # Pretty-print with indentation
        formatted = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)
        return f"JSON Content:\n{formatted}"
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        return f"Error: Invalid JSON format - {str(e)}"
    except Exception as e:
        logger.error(f"Failed to parse JSON file {file_path}: {e}")
        return f"Error reading JSON file: {str(e)}"


def extract_csv(file_path: str) -> str:
    """
    Read CSV file and format as readable text.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        CSV content formatted as readable text
    """
    try:
        encoding = _detect_encoding(file_path)
        
        with open(file_path, 'r', encoding=encoding, errors='replace', newline='') as f:
            # Detect CSV dialect
            sample = f.read(1024)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
            except csv.Error:
                dialect = csv.excel  # Default dialect
            
            reader = csv.reader(f, dialect)
            rows = []
            row_count = 0
            
            for row in reader:
                if row_count >= MAX_CSV_ROWS:
                    rows.append([f"... truncated at {MAX_CSV_ROWS} rows"])
                    break
                rows.append(row)
                row_count += 1
            
            if not rows:
                return "Empty CSV file"
            
            # Format as readable text
            result = ["CSV Content:"]
            
            # Add header if it looks like one
            if rows and len(rows) > 1:
                header = rows[0]
                result.append(f"Headers: {', '.join(header)}")
                result.append("=" * 50)
                data_rows = rows[1:]
            else:
                data_rows = rows
            
            # Add data rows
            for i, row in enumerate(data_rows, 1):
                result.append(f"Row {i}: {', '.join(str(cell) for cell in row)}")
            
            return '\n'.join(result)
            
    except Exception as e:
        logger.error(f"Failed to parse CSV file {file_path}: {e}")
        return f"Error reading CSV file: {str(e)}"


def extract_yaml(file_path: str) -> str:
    """
    Parse and pretty-print YAML file.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        YAML content formatted as string
    """
    if not HAS_YAML:
        logger.info(f"PyYAML not available, reading {file_path} as plain text")
        return extract_plaintext(file_path)
    
    try:
        encoding = _detect_encoding(file_path)
        
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()
        
        # Handle multi-document YAML
        documents = []
        try:
            for doc in yaml.safe_load_all(content):
                if doc is not None:
                    documents.append(doc)
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {file_path}: {e}")
            return f"Error: Invalid YAML format - {str(e)}"
        
        if not documents:
            return "Empty YAML file"
        
        # Format output
        result = ["YAML Content:"]
        
        if len(documents) == 1:
            result.append(yaml.dump(documents[0], default_flow_style=False, sort_keys=True))
        else:
            for i, doc in enumerate(documents, 1):
                result.append(f"\n--- Document {i} ---")
                result.append(yaml.dump(doc, default_flow_style=False, sort_keys=True))
        
        return '\n'.join(result)
        
    except Exception as e:
        logger.error(f"Failed to parse YAML file {file_path}: {e}")
        return f"Error reading YAML file: {str(e)}"


def extract_docx(file_path: str) -> str:
    """
    Extract paragraph text from DOCX file.
    
    Args:
        file_path: Path to the DOCX file
        
    Returns:
        Extracted text from all paragraphs
    """
    if not HAS_DOCX:
        return "DOCX extraction requires python-docx. Install with: pip install python-docx"
    
    try:
        doc = docx.Document(file_path)
        
        paragraphs = []
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:  # Only add non-empty paragraphs
                paragraphs.append(text)
        
        if not paragraphs:
            return "No text content found in DOCX file"
        
        return '\n\n'.join(paragraphs)
        
    except Exception as e:
        logger.error(f"Failed to extract text from DOCX file {file_path}: {e}")
        return f"Error reading DOCX file: {str(e)}"