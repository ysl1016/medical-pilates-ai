import torch
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm
from korean_utils import KoreanTextProcessor

@dataclass
class Document:
    """문서 데이터 클래스"""
    content: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None

class RAGProcessor:
    def __init__(self, 
                 model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 generator_name: str = "google/flan-t5-large",
                 device: str = None):
        """
        RAG 프로세서 초기화
        Args:
            model_name: 임베딩 모델 이름
            generator_name: 생성 모델 이름
            device: 사용할 디바이스 (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 임베딩 모델 초기화
        self.embedding_model = SentenceTransformer(model_name, device=self.device)
        
        # 생성 모델 초기화
        self.generator_tokenizer = AutoTokenizer.from_pretrained(generator_name)
        self.generator_model = AutoModelForSeq2SeqLM.from_pretrained(generator_name).to(self.device)
        
        # FAISS 인덱스 초기화
        self.index = None
        self.documents = []
        
    def add_textbook_content(self, content: Dict[str, str], chunk_size: int = 500):
        """
        교재 내용을 청크로 분할하여 추가
        """
        for section, text in content.items():
            chunks = self._split_text(text, chunk_size)
            for chunk in chunks:
                doc = Document(
                    content=chunk,
                    metadata={"source": "textbook", "section": section}
                )
                self.documents.append(doc)
    
    def add_medical_paper(self, paper: Dict[str, str]):
        """
        의학 논문 추가
        """
        # 논문의 주요 섹션별로 처리
        sections = ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]
        for section in sections:
            if section in paper:
                doc = Document(
                    content=paper[section],
                    metadata={
                        "source": "paper",
                        "title": paper.get("title", ""),
                        "authors": paper.get("authors", []),
                        "year": paper.get("year", ""),
                        "section": section
                    }
                )
                self.documents.append(doc)
    
    def build_index(self):
        """FAISS 인덱스 구축"""
        print("임베딩 생성 중...")
        embeddings = []
        for doc in tqdm(self.documents):
            embedding = self.embedding_model.encode(doc.content)
            doc.embedding = embedding
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        dimension = embeddings.shape[1]
        
        # FAISS 인덱스 생성
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        print(f"인덱스 구축 완료: {len(self.documents)} 문서")
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        쿼리와 관련된 문서 검색
        """
        query_embedding = self.embedding_model.encode(query)
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1),
            k
        )
        
        return [self.documents[i] for i in indices[0]]
    
    def generate_evidence_based_prescription(self, 
                                          patient_info: str,
                                          condition: str,
                                          retrieved_docs: List[Document]) -> Dict[str, str]:
        """
        검색된 문서를 기반으로 근거 기반 처방 생성
        """
        # 컨텍스트 구성
        context = self._prepare_context(retrieved_docs)
        
        # 프롬프트 구성
        prompt = f"""
        Based on the following patient information and medical evidence, 
        create a detailed Medical Pilates prescription:

        Patient Information:
        {patient_info}

        Condition:
        {condition}

        Medical Evidence and Guidelines:
        {context}

        Generate a prescription that includes:
        1. Evidence-based rationale
        2. Specific exercise recommendations
        3. Scientific references
        4. Safety considerations
        """
        
        # 처방 생성
        inputs = self.generator_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.generator_model.generate(
            **inputs,
            max_length=1500,
            num_beams=4,
            temperature=0.7,
            no_repeat_ngram_size=3
        )
        
        prescription = self.generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 참조 문헌 정보 수집
        references = self._collect_references(retrieved_docs)
        
        return {
            "prescription": prescription,
            "references": references
        }
    
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """텍스트를 청크로 분할"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _prepare_context(self, documents: List[Document]) -> str:
        """검색된 문서들을 컨텍스트로 구성"""
        context_parts = []
        
        for doc in documents:
            if doc.metadata["source"] == "textbook":
                context_parts.append(f"Textbook ({doc.metadata['section']}):\n{doc.content}")
            else:  # paper
                context_parts.append(
                    f"Research Paper ({doc.metadata['year']}, {doc.metadata['section']}):\n{doc.content}"
                )
        
        return "\n\n".join(context_parts)
    
    def _collect_references(self, documents: List[Document]) -> List[Dict]:
        """참조 문헌 정보 수집"""
        references = []
        
        for doc in documents:
            if doc.metadata["source"] == "paper":
                ref = {
                    "title": doc.metadata["title"],
                    "authors": doc.metadata["authors"],
                    "year": doc.metadata["year"],
                    "section": doc.metadata["section"]
                }
                if ref not in references:
                    references.append(ref)
        
        return references
    
    def save_index(self, path: str):
        """인덱스 및 문서 저장"""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # FAISS 인덱스 저장
        faiss.write_index(self.index, str(save_dir / "index.faiss"))
        
        # 문서 데이터 저장
        docs_data = []
        for doc in self.documents:
            doc_dict = {
                "content": doc.content,
                "metadata": doc.metadata,
                "embedding": doc.embedding.tolist() if doc.embedding is not None else None
            }
            docs_data.append(doc_dict)
        
        with open(save_dir / "documents.json", "w", encoding="utf-8") as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)
    
    def load_index(self, path: str):
        """인덱스 및 문서 로드"""
        load_dir = Path(path)
        
        # FAISS 인덱스 로드
        self.index = faiss.read_index(str(load_dir / "index.faiss"))
        
        # 문서 데이터 로드
        with open(load_dir / "documents.json", "r", encoding="utf-8") as f:
            docs_data = json.load(f)
        
        self.documents = []
        for doc_dict in docs_data:
            doc = Document(
                content=doc_dict["content"],
                metadata=doc_dict["metadata"]
            )
            if doc_dict["embedding"] is not None:
                doc.embedding = np.array(doc_dict["embedding"])
            self.documents.append(doc)

class RAGSystem:
    """RAG 기반 질의응답 시스템"""
    
    def __init__(self, processor=None, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        """RAG 시스템 초기화"""
        self.processor = processor or RAGProcessor(model_name=model_name)
        self.korean_processor = KoreanTextProcessor()
    
    def add_knowledge_base(self, textbooks=None, papers=None, exercises=None):
        """지식 베이스 추가"""
        if textbooks:
            for textbook in textbooks:
                self.processor.add_textbook_content(textbook)
                
        if papers:
            for paper in papers:
                self.processor.add_medical_paper(paper)
        
        # 인덱스 구축
        self.processor.build_index()
    
    def generate_response(self, query, context=None):
        """사용자 질의에 대한 응답 생성"""
        # 한국어 질의 처리
        query_ko = query
        query_en = self.korean_processor.translate_ko_to_en(query)
        
        # 컨텍스트 정보 준비
        if context:
            patient_info = f"""
            통증 수준: {context.get('pain_level', 0)}/10
            경험 수준: {context.get('experience_level', '초보자')}
            """
        else:
            patient_info = "일반적인 건강 상태"
            
        # 관련 문서 검색
        retrieved_docs = self.processor.retrieve(query_en, k=5)
        
        # 응답 생성
        result = self.processor.generate_evidence_based_prescription(
            patient_info=patient_info,
            condition=query_en,
            retrieved_docs=retrieved_docs
        )
        
        # 한국어 응답 변환
        response_ko = self.korean_processor.translate_en_to_ko(result['prescription'])
        
        return response_ko