import fitz  # PyMuPDF
from docx import Document
import pandas as pd
import os
from typing import List, Dict, Union, Generator
import gc

class DocumentParser:
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.xlsx', '.csv']
        
    def parse_pdf(self, file_path: str, batch_size: int = 5) -> Generator[List[str], None, None]:
        """PDF 파일을 배치 단위로 처리하여 메모리 사용을 최적화합니다."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        doc = fitz.open(file_path)
        current_batch = []
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                current_batch.append(text)
                
                if len(current_batch) >= batch_size:
                    yield current_batch
                    current_batch = []
                    gc.collect()  # 메모리 정리
            
            # 남은 페이지 처리
            if current_batch:
                yield current_batch
                
        finally:
            doc.close()
            gc.collect()
    
    def parse_docx(self, file_path: str, batch_size: int = 5) -> Generator[List[str], None, None]:
        """DOCX 파일을 배치 단위로 처리합니다."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        doc = Document(file_path)
        current_batch = []
        
        try:
            for para in doc.paragraphs:
                if para.text.strip():
                    current_batch.append(para.text)
                    
                if len(current_batch) >= batch_size:
                    yield current_batch
                    current_batch = []
                    gc.collect()
            
            if current_batch:
                yield current_batch
                
        finally:
            gc.collect()
    
    def parse_excel(self, file_path: str, batch_size: int = 1000) -> Generator[List[str], None, None]:
        """엑셀 파일을 배치 단위로 처리합니다."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        for chunk in pd.read_excel(file_path, chunksize=batch_size):
            yield chunk.values.tolist()
            gc.collect()
    
    def parse_csv(self, file_path: str, batch_size: int = 1000) -> Generator[List[str], None, None]:
        """CSV 파일을 배치 단위로 처리합니다."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        for chunk in pd.read_csv(file_path, chunksize=batch_size):
            yield chunk.values.tolist()
            gc.collect()
    
    def parse_document(self, file_path: str, batch_size: int = 5) -> Generator[List[str], None, None]:
        """파일 형식에 따라 적절한 파서를 선택하여 처리합니다."""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {ext}")
        
        if ext == '.pdf':
            yield from self.parse_pdf(file_path, batch_size)
        elif ext == '.docx':
            yield from self.parse_docx(file_path, batch_size)
        elif ext == '.xlsx':
            yield from self.parse_excel(file_path, batch_size)
        elif ext == '.csv':
            yield from self.parse_csv(file_path, batch_size)