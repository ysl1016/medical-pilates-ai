import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from korean_utils import KoreanTextProcessor
import gc

class RAGSystem:
    def __init__(self, model_name="jhgan/ko-sroberta-multitask", device=None):
        self.korean_processor = KoreanTextProcessor()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 로드를 지연시킴
        self._sentence_transformer = None
        self._tokenizer = None
        self._model = None
        
        self.model_name = model_name
        self.faiss_index = None
        self.documents = []
        
    @property
    def sentence_transformer(self):
        if self._sentence_transformer is None:
            self._sentence_transformer = SentenceTransformer(self.model_name).to(self.device)
        return self._sentence_transformer
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained("beomi/KoAlpaca-Polyglot-12.8B")
        return self._tokenizer
    
    @property
    def model(self):
        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained("beomi/KoAlpaca-Polyglot-12.8B").to(self.device)
        return self._model
        
    def add_documents(self, documents: list, batch_size: int = 32):
        """문서를 배치 단위로 처리하여 RAG 시스템에 추가합니다."""
        self.documents.extend(documents)
        
        # 배치 단위로 임베딩 생성
        embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_embeddings = self.sentence_transformer.encode(batch)
            embeddings.append(batch_embeddings)
            
            # 메모리 정리
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 모든 임베딩 합치기
        all_embeddings = np.vstack(embeddings)
        
        # FAISS 인덱스 업데이트
        self._update_faiss_index(all_embeddings)
        
        # 메모리 정리
        del embeddings, all_embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _update_faiss_index(self, embeddings):
        """FAISS 인덱스를 업데이트합니다."""
        if self.faiss_index is None:
            self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.faiss_index.add(embeddings)
    
    def search_documents(self, query: str, k: int = 3) -> list:
        """쿼리와 관련된 문서를 검색합니다."""
        query_embedding = self.sentence_transformer.encode([query])
        D, I = self.faiss_index.search(query_embedding, k)
        
        # 메모리 정리
        del query_embedding
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return [self.documents[i] for i in I[0]]
    
    def generate_response(self, query: str, context: dict = None) -> str:
        """사용자 쿼리에 대한 응답을 생성합니다."""
        try:
            relevant_docs = self.search_documents(query)
            prompt = self._create_prompt(query, relevant_docs, context)
            response = self._generate_text(prompt)
            
            return self.korean_processor.post_process_response(response)
            
        finally:
            # 메모리 정리
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _create_prompt(self, query: str, relevant_docs: list, context: dict = None) -> str:
        """프롬프트를 생성합니다."""
        base_prompt = f"다음 정보를 바탕으로 질문에 답변해주세요:\n\n"
        for doc in relevant_docs:
            base_prompt += f"참고 문서: {doc}\n"
        
        if context:
            base_prompt += f"\n상황 정보:\n"
            for key, value in context.items():
                base_prompt += f"- {key}: {value}\n"
        
        base_prompt += f"\n질문: {query}\n답변:"
        return base_prompt
    
    def _generate_text(self, prompt: str) -> str:
        """텍스트를 생성합니다."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        finally:
            # 메모리 정리
            del inputs, outputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def clear_memory(self):
        """메모리를 정리합니다."""
        if self._sentence_transformer is not None:
            del self._sentence_transformer
            self._sentence_transformer = None
            
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
            
        if self._model is not None:
            del self._model
            self._model = None
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()