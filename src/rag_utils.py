class RAGSystem:
    def __init__(self):
        self.categories = {
            'basic': ['기초', '입문', '기본'],
            'intermediate': ['중급', '심화'],
            'advanced': ['고급', '마스터'],
            'rehabilitation': ['재활', '치료', '통증'],
            'anatomy': ['해부학', '근육', '골격'],
            'special_conditions': ['임산부', '노인', '특수조건']
        }
        self.document_categories = {}  # 문서별 카테고리 저장
        self.category_embeddings = {}  # 카테고리별 임베딩 저장

    def categorize_document(self, content):
        """문서 내용을 분석하여 카테고리 분류"""
        categories = []
        content_text = ' '.join(content) if isinstance(content, list) else content
        
        for category, keywords in self.categories.items():
            if any(keyword in content_text for keyword in keywords):
                categories.append(category)
        
        return categories if categories else ['general']

    def add_documents(self, documents, batch_size=32, source_info=None):
        """카테고리 정보를 포함하여 문서 추가"""
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # 각 문서의 카테고리 분류
            for doc in batch:
                categories = self.categorize_document(doc)
                doc_id = len(self.documents)
                self.document_categories[doc_id] = categories
                
            # 기존의 문서 처리 로직
            embeddings = self.get_embeddings(batch)
            self.index.add(embeddings)
            self.documents.extend(batch)
            
            if source_info:
                self.document_sources.extend([source_info] * len(batch))
            
            # 메모리 정리
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def search_documents(self, query, context=None, top_k=5):
        """컨텍스트와 카테고리를 고려한 문서 검색"""
        query_embedding = self.get_embeddings([query])[0]
        
        # 컨텍스트 기반 카테고리 우선순위 설정
        priority_categories = self._get_priority_categories(context)
        
        # 검색 및 재정렬
        D, I = self.index.search(query_embedding.reshape(1, -1), top_k * 2)
        
        # 카테고리 기반 점수 조정
        scored_results = []
        for idx, (distance, doc_idx) in enumerate(zip(D[0], I[0])):
            doc_categories = self.document_categories.get(doc_idx, ['general'])
            category_bonus = sum(2.0 if cat in priority_categories else 1.0 
                               for cat in doc_categories)
            
            scored_results.append({
                'idx': doc_idx,
                'distance': distance,
                'adjusted_score': distance * (1.0 / category_bonus),
                'categories': doc_categories
            })
        
        # 조정된 점수로 재정렬
        scored_results.sort(key=lambda x: x['adjusted_score'])
        
        # 상위 K개 결과 반환
        top_results = scored_results[:top_k]
        return [self.documents[r['idx']] for r in top_results]

    def _get_priority_categories(self, context):
        """컨텍스트 기반 우선순위 카테고리 결정"""
        if not context:
            return []
            
        priority_categories = []
        
        # 경험 수준에 따른 카테고리
        experience_level = context.get('experience_level', '').lower()
        if experience_level == 'beginner':
            priority_categories.extend(['basic', 'rehabilitation'])
        elif experience_level == 'intermediate':
            priority_categories.append('intermediate')
        elif experience_level == 'advanced':
            priority_categories.append('advanced')
            
        # 통증 수준에 따른 카테고리
        pain_level = context.get('pain_level', '').lower()
        if pain_level in ['moderate', 'severe']:
            priority_categories.extend(['rehabilitation', 'anatomy'])
            
        # 특수 조건 확인
        conditions = context.get('special_conditions', [])
        if conditions:
            priority_categories.append('special_conditions')
            
        return priority_categories

    def generate_response(self, query, context=None):
        """컨텍스트 기반 개선된 응답 생성"""
        relevant_docs = self.search_documents(query, context)
        
        # 컨텍스트 기반 프롬프트 생성
        prompt = self._create_context_aware_prompt(query, relevant_docs, context)
        
        try:
            response = self.generate_text(prompt)
            return self._format_response(response, context)
        except Exception as e:
            print(f"Error generating response: {e}")
            return "죄송합니다. 응답 생성 중 오류가 발생했습니다."

    def _create_context_aware_prompt(self, query, relevant_docs, context):
        """컨텍스트를 고려한 프롬프트 생성"""
        context_str = ""
        if context:
            if 'experience_level' in context:
                context_str += f"\n경험 수준: {context['experience_level']}"
            if 'pain_level' in context:
                context_str += f"\n통증 수준: {context['pain_level']}"
            if 'special_conditions' in context:
                context_str += f"\n특수 조건: {', '.join(context['special_conditions'])}"
        
        docs_text = "\n".join(relevant_docs)
        
        prompt = f"""
다음 정보를 바탕으로 질문에 답변해주세요:

사용자 정보:{context_str}

관련 문서 내용:
{docs_text}

질문: {query}

답변 시 다음 사항을 고려해주세요:
1. 사용자의 경험 수준과 신체 상태
2. 안전 주의사항
3. 단계별 진행 방법
4. 필요한 경우 대체 운동 제시
"""
        return prompt

    def _format_response(self, response, context):
        """컨텍스트에 맞게 응답 포맷팅"""
        experience_level = context.get('experience_level', '').lower()
        pain_level = context.get('pain_level', '').lower()
        
        # 경험 수준별 추가 정보
        if experience_level == 'beginner':
            response += "\n\n초보자를 위한 추가 팁:\n- 천천히 진행하세요\n- 불편함을 느끼면 즉시 중단하세요"
        
        # 통증 수준별 주의사항
        if pain_level in ['moderate', 'severe']:
            response += "\n\n주의사항:\n- 통증이 심해지면 즉시 중단하세요\n- 필요한 경우 전문가와 상담하세요"
        
        return response