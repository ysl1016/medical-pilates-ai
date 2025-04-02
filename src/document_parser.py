import os
from pathlib import Path
import json
import re
from typing import Dict, List, Union, Optional
import fitz  # PyMuPDF for PDF parsing
import markdown
import bs4
from docx import Document
from datetime import datetime
import numpy as np
from langdetect import detect
import logging

class DocumentParser:
    """다양한 형식의 문서를 파싱하여 JSON으로 변환하는 클래스"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.setup_logging()
        
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('document_parsing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def parse_directory(self, input_dir: str, output_dir: str) -> List[Dict]:
        """디렉토리 내의 모든 지원 파일 파싱"""
        input_path = self.base_path / input_dir
        output_path = self.base_path / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        parsed_documents = []
        
        for file_path in input_path.rglob("*"):
            if not file_path.is_file():
                continue
                
            try:
                if file_path.suffix.lower() == '.pdf':
                    doc_data = self.parse_pdf(file_path)
                elif file_path.suffix.lower() == '.md':
                    doc_data = self.parse_markdown(file_path)
                elif file_path.suffix.lower() in ['.docx', '.doc']:
                    doc_data = self.parse_docx(file_path)
                else:
                    continue
                
                if doc_data:
                    # 파일명에서 날짜 추출 (예: filename_YYMMDD.pdf)
                    date_match = re.search(r'_(\d{6})', file_path.stem)
                    if date_match:
                        doc_data['metadata']['date'] = date_match.group(1)
                    
                    # 언어 감지
                    doc_data['metadata']['language'] = self.detect_language(doc_data['content'])
                    
                    # JSON 파일로 저장
                    output_file = output_path / f"{file_path.stem}.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(doc_data, f, ensure_ascii=False, indent=2)
                    
                    parsed_documents.append(doc_data)
                    self.logger.info(f"Successfully parsed and saved: {file_path.name}")
                    
            except Exception as e:
                self.logger.error(f"Error parsing {file_path.name}: {str(e)}")
                
        return parsed_documents

    def parse_pdf(self, file_path: Path) -> Optional[Dict]:
        """PDF 파일 파싱"""
        try:
            doc = fitz.open(file_path)
            content = ""
            toc = []
            
            # 목차 추출
            pdf_toc = doc.get_toc()
            if pdf_toc:
                for t in pdf_toc:
                    toc.append({
                        "level": t[0],
                        "title": t[1],
                        "page": t[2]
                    })
            
            # 섹션별 내용 추출
            current_section = {"title": "Main Content", "content": ""}
            sections = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # 섹션 제목 감지 (대문자로 된 줄이나 번호가 있는 줄)
                lines = text.split('\n')
                for line in lines:
                    if re.match(r'^[0-9]+\.|\s*[A-Z\s]{4,}$', line.strip()):
                        if current_section["content"].strip():
                            sections.append(current_section)
                        current_section = {"title": line.strip(), "content": ""}
                    else:
                        current_section["content"] += line + "\n"
            
            if current_section["content"].strip():
                sections.append(current_section)
            
            # 메타데이터 추출
            metadata = {
                "title": file_path.stem,
                "pages": len(doc),
                "toc": toc,
                "source_type": "pdf",
                "creation_date": datetime.now().strftime("%Y-%m-%d"),
                "filename": file_path.name
            }
            
            return {
                "metadata": metadata,
                "sections": sections,
                "content": "\n".join(section["content"] for section in sections)
            }
            
        except Exception as e:
            self.logger.error(f"PDF parsing error for {file_path.name}: {str(e)}")
            return None

    def parse_markdown(self, file_path: Path) -> Optional[Dict]:
        """Markdown 파일 파싱"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            # Markdown을 HTML로 변환
            html_content = markdown.markdown(md_content)
            soup = bs4.BeautifulSoup(html_content, 'html.parser')
            
            # 섹션 추출
            sections = []
            current_section = {"title": "Introduction", "content": ""}
            
            for elem in soup.find_all(['h1', 'h2', 'h3', 'p']):
                if elem.name in ['h1', 'h2', 'h3']:
                    if current_section["content"].strip():
                        sections.append(current_section)
                    current_section = {"title": elem.text.strip(), "content": ""}
                else:
                    current_section["content"] += elem.text + "\n"
            
            if current_section["content"].strip():
                sections.append(current_section)
            
            metadata = {
                "title": file_path.stem,
                "source_type": "markdown",
                "creation_date": datetime.now().strftime("%Y-%m-%d"),
                "filename": file_path.name
            }
            
            return {
                "metadata": metadata,
                "sections": sections,
                "content": "\n".join(section["content"] for section in sections)
            }
            
        except Exception as e:
            self.logger.error(f"Markdown parsing error for {file_path.name}: {str(e)}")
            return None

    def parse_docx(self, file_path: Path) -> Optional[Dict]:
        """DOCX 파일 파싱"""
        try:
            doc = Document(file_path)
            sections = []
            current_section = {"title": "Main Content", "content": ""}
            
            for para in doc.paragraphs:
                # 제목 스타일이나 대문자로 된 짧은 텍스트를 섹션 제목으로 처리
                if para.style.name.startswith('Heading') or (
                    len(para.text.strip()) < 100 and para.text.isupper()):
                    if current_section["content"].strip():
                        sections.append(current_section)
                    current_section = {"title": para.text.strip(), "content": ""}
                else:
                    current_section["content"] += para.text + "\n"
            
            if current_section["content"].strip():
                sections.append(current_section)
            
            metadata = {
                "title": file_path.stem,
                "source_type": "docx",
                "creation_date": datetime.now().strftime("%Y-%m-%d"),
                "filename": file_path.name
            }
            
            return {
                "metadata": metadata,
                "sections": sections,
                "content": "\n".join(section["content"] for section in sections)
            }
            
        except Exception as e:
            self.logger.error(f"DOCX parsing error for {file_path.name}: {str(e)}")
            return None

    def detect_language(self, text: str) -> str:
        """텍스트 언어 감지"""
        try:
            return detect(text[:1000])  # 첫 1000자만 사용하여 언어 감지
        except:
            return "unknown"

    def extract_exercise_info(self, content: str) -> Dict:
        """운동 관련 정보 추출"""
        exercise_info = {
            "name": None,
            "equipment": [],
            "difficulty": None,
            "target_muscles": [],
            "contraindications": [],
            "modifications": [],
            "instructions": []
        }
        
        # 정규표현식 패턴
        patterns = {
            "name": r"운동명[:\s]+(.*?)(?=\n|$)",
            "equipment": r"장비[:\s]+(.*?)(?=\n|$)",
            "difficulty": r"난이도[:\s]+(초급|중급|고급)",
            "target_muscles": r"목표\s*근육[:\s]+(.*?)(?=\n|$)",
            "contraindications": r"금기사항[:\s]+(.*?)(?=\n|$)",
            "modifications": r"수정[/변형]*\s*사항[:\s]+(.*?)(?=\n|$)",
            "instructions": r"수행\s*방법[:\s]+(.*?)(?=\n|$)"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if match:
                value = match.group(1).strip()
                if key in ["equipment", "target_muscles", "contraindications", "modifications"]:
                    exercise_info[key] = [item.strip() for item in value.split(',')]
                else:
                    exercise_info[key] = value
        
        return exercise_info

    def process_medical_papers(self, papers_data: List[Dict]) -> List[Dict]:
        """의학 논문 데이터 처리"""
        processed_papers = []
        
        for paper in papers_data:
            try:
                # 논문 메타데이터 추출
                metadata = {
                    "title": paper.get("title", ""),
                    "authors": paper.get("authors", []),
                    "year": paper.get("year", ""),
                    "journal": paper.get("journal", ""),
                    "doi": paper.get("doi", ""),
                    "source_type": "medical_paper"
                }
                
                # 섹션별 내용 처리
                sections = []
                for section_name in ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]:
                    if section_name in paper:
                        sections.append({
                            "title": section_name.capitalize(),
                            "content": paper[section_name]
                        })
                
                processed_paper = {
                    "metadata": metadata,
                    "sections": sections,
                    "content": "\n\n".join(section["content"] for section in sections)
                }
                
                processed_papers.append(processed_paper)
                
            except Exception as e:
                self.logger.error(f"Error processing paper {paper.get('title', 'Unknown')}: {str(e)}")
        
        return processed_papers