"""
RAG (Retrieval-Augmented Generation) System for Medical Pilates AI
Handles document retrieval and response generation
"""

import os
import json
import torch
import faiss
import numpy as np
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from korean_utils import KoreanTextProcessor

class RAGSystem:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.korean_processor = KoreanTextProcessor()
        self.index = None
        self.documents = []
        self.document_embeddings = None
        
        # LLM 설정
        self.tokenizer = AutoTokenizer.from_pretrained(
            "beomi/KoAlpaca-Polyglot-12.8B",
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "beomi/KoAlpaca-Polyglot-12.8B",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto"
        )
        
    def add_documents(self, documents: List[Dict]):
        """문서 추가 및 임베딩 생성"""
        self.documents.extend(documents)
        
        # 문서 텍스트 추출 및 전처리
        texts = [doc.get('content', '') for doc in documents]
        texts = [self.korean_processor.preprocess_text(text) for text in texts]
        
        # 임베딩 생성
        new_embeddings = self.embedding_model.encode(texts)
        
        if self.document_embeddings is None:
            self.document_embeddings = new_embeddings
        else:
            self.document_embeddings = np.vstack([self.document_embeddings, new_embeddings])
            
        # FAISS 인덱스 업데이트
        self._update_index()
        
    def _update_index(self):
        """FAISS 인덱스 업데이트"""
        dimension = self.document_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.document_embeddings.astype('float32'))
        
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """쿼리에 대한 관련 문서 검색"""
        # 쿼리 전처리 및 임베딩
        query = self.korean_processor.preprocess_text(query)
        query_embedding = self.embedding_model.encode([query])[0]
        
        # 유사도 검색
        D, I = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            k
        )
        
        # 검색 결과 반환
        results = []
        for idx, (distance, doc_idx) in enumerate(zip(D[0], I[0])):
            if doc_idx < len(self.documents):
                doc = self.documents[doc_idx].copy()
                doc['score'] = float(1 / (1 + distance))  # 거리를 점수로 변환
                results.append(doc)
                
        return results
    
    def generate_response(self, query: str, context: str) -> str:
        """컨텍스트를 기반으로 응답 생성"""
        prompt = f"""아래는 필라테스 운동과 관련된 정보입니다:

{context}

질문: {query}

위 정보를 바탕으로 답변해주세요. 정보가 불충분하다면, 일반적인 필라테스 원칙에 기반하여 답변해주세요.

답변:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        return response
    
    def process_query(self, query: str, k: int = 3) -> str:
        """쿼리 처리 및 응답 생성"""
        # 관련 문서 검색
        relevant_docs = self.search(query, k=k)
        
        # 컨텍스트 구성
        context = "\n\n".join([
            f"문서 {i+1}:\n{doc.get('content', '')}"
            for i, doc in enumerate(relevant_docs)
        ])
        
        # 응답 생성
        response = self.generate_response(query, context)
        
        return response
    
    def analyze_exercise(self, exercise_text: str) -> Dict:
        """운동 분석"""
        # 기본 정보 추출
        components = self.korean_processor.extract_exercise_components(exercise_text)
        
        # 난이도 분석
        difficulty = self.korean_processor.analyze_difficulty(exercise_text)
        
        # 운동 초점 분석
        focus = self.korean_processor.analyze_exercise_focus(exercise_text)
        
        # 키워드 추출
        keywords = self.korean_processor.extract_keywords(exercise_text)
        
        return {
            'components': components,
            'difficulty': difficulty,
            'focus_areas': focus,
            'keywords': keywords
        }
    
    def generate_exercise_recommendation(self, 
                                      patient_info: Dict,
                                      condition: str,
                                      difficulty_level: str = '기본') -> Dict:
        """환자 정보에 기반한 운동 추천"""
        # 검색 쿼리 구성
        query = f"{condition} {difficulty_level} 운동"
        relevant_docs = self.search(query, k=5)
        
        # 운동 추천 생성
        recommendations = []
        for doc in relevant_docs:
            exercise_analysis = self.analyze_exercise(doc.get('content', ''))
            
            # 환자 상태와 운동 난이도 매칭
            if exercise_analysis['difficulty']['overall_level'] == difficulty_level:
                recommendations.append({
                    'exercise': doc.get('content', ''),
                    'analysis': exercise_analysis,
                    'score': doc.get('score', 0)
                })
        
        # 점수순 정렬
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'patient_info': patient_info,
            'condition': condition,
            'difficulty_level': difficulty_level,
            'recommendations': recommendations[:3]  # 상위 3개 추천
        }
    
    def save_index(self, path: str):
        """인덱스 및 문서 저장"""
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
            
        # FAISS 인덱스 저장
        faiss.write_index(self.index, f"{path}.index")
        
        # 문서 및 임베딩 저장
        np.save(f"{path}_embeddings.npy", self.document_embeddings)
        with open(f"{path}_documents.json", 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
            
    def load_index(self, path: str):
        """인덱스 및 문서 로드"""
        # FAISS 인덱스 로드
        self.index = faiss.read_index(f"{path}.index")
        
        # 문서 및 임베딩 로드
        self.document_embeddings = np.load(f"{path}_embeddings.npy")
        with open(f"{path}_documents.json", 'r', encoding='utf-8') as f:
            self.documents = json.load(f)