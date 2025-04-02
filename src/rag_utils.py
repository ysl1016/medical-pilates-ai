import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from korean_utils import KoreanTextProcessor

class RAGSystem:
    def __init__(self):
        self.korean_processor = KoreanTextProcessor()
        self.sentence_transformer = SentenceTransformer('jhgan/ko-sroberta-multitask')
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/KoAlpaca-Polyglot-12.8B")
        self.model = AutoModelForCausalLM.from_pretrained("beomi/KoAlpaca-Polyglot-12.8B")
        self.faiss_index = None
        self.documents = []
        
    def add_documents(self, documents):
        """문서를 RAG 시스템에 추가합니다."""
        self.documents.extend(documents)
        self._update_faiss_index()
    
    def _update_faiss_index(self):
        """FAISS 인덱스를 업데이트합니다."""
        embeddings = self.sentence_transformer.encode(self.documents)
        self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.faiss_index.add(embeddings)
    
    def search_documents(self, query, k=3):
        """쿼리와 관련된 문서를 검색합니다."""
        query_embedding = self.sentence_transformer.encode([query])
        D, I = self.faiss_index.search(query_embedding, k)
        return [self.documents[i] for i in I[0]]
    
    def generate_response(self, query, context=None):
        """사용자 쿼리에 대한 응답을 생성합니다."""
        relevant_docs = self.search_documents(query)
        prompt = self._create_prompt(query, relevant_docs, context)
        
        response = self._generate_text(prompt)
        return self.korean_processor.post_process_response(response)
    
    def _create_prompt(self, query, relevant_docs, context=None):
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
    
    def _generate_text(self, prompt):
        """텍스트를 생성합니다."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def analyze_exercise(self, exercise_description):
        """운동 설명을 분석하여 난이도와 주의사항을 추출합니다."""
        prompt = f"""다음 운동 설명을 분석하여 난이도와 주의사항을 추출해주세요:

운동 설명: {exercise_description}

다음 형식으로 답변해주세요:
난이도: [초급/중급/고급]
주의사항:
- 항목 1
- 항목 2
- 항목 3"""
        
        response = self._generate_text(prompt)
        return self.korean_processor.extract_exercise_analysis(response)
    
    def generate_exercise_recommendation(self, patient_info):
        """환자 정보를 바탕으로 운동을 추천합니다."""
        prompt = f"""다음 환자 정보를 바탕으로 적절한 필라테스 운동을 추천해주세요:

환자 정보:
{patient_info}

다음 형식으로 답변해주세요:
1. [운동명]
   - 설명: [상세 설명]
   - 난이도: [초급/중급/고급]
   - 주의사항: [주의사항]

2. [운동명]
   ...
"""
        response = self._generate_text(prompt)
        return self.korean_processor.extract_exercise_recommendations(response)