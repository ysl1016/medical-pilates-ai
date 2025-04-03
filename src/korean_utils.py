"""
Korean Text Processing Utilities for Medical Pilates AI
Handles Korean language processing, translation, and analysis
"""

import re
import logging
from typing import Dict, List, Optional
from transformers import pipeline
from konlpy.tag import Mecab
import numpy as np

class KoreanTextProcessor:
    def __init__(self):
        """
        한국어 텍스트 처리를 위한 클래스 초기화
        """
        # 한국어 처리를 위한 모델 초기화
        self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en")
        self.ko_summarizer = pipeline("summarization", model="digit82/kobart-summarization")
        self.TRANSLATION_BATCH_SIZE = 8  # 배치 처리 크기 설정
        
        try:
            self.mecab = Mecab()
        except Exception as e:
            logging.warning(f"Mecab initialization failed: {str(e)}. Using basic text processing.")
            self.mecab = None
        
        self.exercise_keywords = {
            '기본': ['기초', '기본', '입문', '초급'],
            '중급': ['중급', '중간', '심화'],
            '고급': ['고급', '상급', '전문가'],
            '통증': ['통증', '아픔', '불편', '불편감'],
            '강화': ['강화', '단련', '향상', '발달'],
            '스트레칭': ['스트레칭', '신장', '늘리기', '유연성'],
            '자세': ['자세', '포지션', '포스처', '정렬']
        }
        
    def translate_ko_to_en(self, text: str) -> str:
        """
        한국어 텍스트를 영어로 번역 - 배치 처리 적용
        """
        # 빈 텍스트 처리
        if not text or not text.strip():
            return ""
            
        # 긴 텍스트를 적절한 크기의 청크로 분할
        max_chunk_size = 1000  # 1000자 단위로 분할
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # 배치 단위로 번역
        batch_size = self.TRANSLATION_BATCH_SIZE
        translated_parts = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            try:
                translations = self.translator(batch, max_length=512)
                if isinstance(translations, list) and isinstance(translations[0], dict):
                    translated_batch = [t['translation_text'] for t in translations]
                else:
                    # 단일 항목인 경우
                    translated_batch = [translations['translation_text']]
                translated_parts.extend(translated_batch)
            except Exception as e:
                logging.error(f"Translation error for batch {i//batch_size}: {str(e)}")
                # 오류 발생 시 원본 반환
                translated_parts.extend(batch)
        
        return ' '.join(translated_parts)
    
    def translate_en_to_ko(self, text: str) -> str:
        """
        영어 텍스트를 한국어로 번역 - 배치 처리 적용
        """
        # 빈 텍스트 처리
        if not text or not text.strip():
            return ""
            
        # 한국어 번역 모델 사용
        translator_ko = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ko")
        
        # 긴 텍스트를 청크로 분할
        max_chunk_size = 1000
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # 배치 단위로 번역
        batch_size = self.TRANSLATION_BATCH_SIZE
        translated_parts = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            try:
                translations = translator_ko(batch, max_length=512)
                if isinstance(translations, list) and isinstance(translations[0], dict):
                    translated_batch = [t['translation_text'] for t in translations]
                else:
                    # 단일 항목인 경우
                    translated_batch = [translations['translation_text']]
                translated_parts.extend(translated_batch)
            except Exception as e:
                logging.error(f"Translation error for batch {i//batch_size}: {str(e)}")
                # 오류 발생 시 원본 반환
                translated_parts.extend(batch)
        
        return ' '.join(translated_parts)
    
    def summarize_korean(self, text: str, max_length: int = 512) -> str:
        """
        한국어 텍스트 요약 - 오류 처리 개선
        """
        if not text or not text.strip():
            return ""
            
        try:
            # 너무 긴 텍스트 처리
            if len(text) > 10000:
                # 청크로 나누어 요약
                chunks = [text[i:i+10000] for i in range(0, len(text), 10000)]
                summaries = []
                for chunk in chunks:
                    try:
                        summary = self.ko_summarizer(chunk, max_length=max_length//len(chunks), 
                                                    min_length=max_length//(len(chunks)*4))[0]['summary_text']
                        summaries.append(summary)
                    except Exception as e:
                        logging.error(f"Summarization error for chunk: {str(e)}")
                        # 오류 발생 시 청크의 앞부분 사용
                        summaries.append(chunk[:max_length//len(chunks)])
                
                return " ".join(summaries)
            else:
                summary = self.ko_summarizer(text, max_length=max_length, min_length=max_length//4)[0]['summary_text']
                return summary
        except Exception as e:
            logging.error(f"Summarization error: {str(e)}")
            # 오류 발생 시 원본 텍스트 일부 반환
            return text[:max_length] + "..."
    
    @staticmethod
    def normalize_korean_text(text: str) -> str:
        """
        한국어 텍스트 정규화
        """
        # 빈 텍스트 처리
        if not text or not text.strip():
            return ""
            
        # 불필요한 공백 제거
        text = re.sub(r'\s+', ' ', text)
        # 특수문자 처리 (일부 특수문자는 유지)
        text = re.sub(r'[^\w\s\.,!?]', '', text)
        return text.strip()
    
    @staticmethod
    def extract_korean_terms(text: str) -> List[str]:
        """
        한국어 용어 추출
        """
        # 빈 텍스트 처리
        if not text or not text.strip():
            return []
            
        # 한글 단어 추출 (2글자 이상)
        korean_terms = re.findall(r'[가-힣]{2,}', text)
        return list(set(korean_terms))
    
    def process_exercise_description(self, description: Dict) -> Dict:
        """
        운동 설명 데이터의 이중 언어 처리
        """
        if not description:
            return {}
            
        bilingual_description = {}
        
        for key, value in description.items():
            if isinstance(value, str) and value.strip():
                # 한국어 원본 저장
                bilingual_description[f"{key}_ko"] = value
                # 영어 번역 추가
                bilingual_description[f"{key}_en"] = self.translate_ko_to_en(value)
            elif isinstance(value, list):
                # 리스트 항목 처리
                bilingual_description[f"{key}_ko"] = value
                bilingual_description[f"{key}_en"] = [
                    self.translate_ko_to_en(item) for item in value if item and item.strip()
                ]
            else:
                bilingual_description[key] = value
                
        return bilingual_description
    
    def create_exercise_summary(self, exercise_data: Dict) -> str:
        """
        운동 데이터에서 한국어 요약 생성
        """
        if not exercise_data:
            return ""
            
        summary_text = f"""
        운동명: {exercise_data.get('name_ko', '')}
        
        주요 목적:
        {exercise_data.get('description_ko', '')}
        
        주의사항:
        {', '.join(exercise_data.get('precautions_ko', []))}
        
        적응증:
        {', '.join(exercise_data.get('indications_ko', []))}
        
        난이도: {exercise_data.get('difficulty_level_ko', '')}
        """
        
        return self.summarize_korean(summary_text)

    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        # 빈 텍스트 처리
        if not text or not text.strip():
            return ""
            
        # 기본 전처리
        text = re.sub(r'\s+', ' ', text)  # 연속된 공백 제거
        text = text.strip()
        
        return text

    def extract_keywords(self, text: str) -> List[str]:
        """핵심 키워드 추출"""
        # 빈 텍스트 처리
        if not text or not text.strip():
            return []
            
        if self.mecab:
            try:
                # Mecab을 사용한 형태소 분석
                morphs = self.mecab.pos(text)
                keywords = [word for word, pos in morphs 
                          if pos in ['NNG', 'NNP', 'VV', 'VA']]
            except Exception as e:
                logging.error(f"Mecab processing error: {str(e)}")
                # 실패 시 기본 키워드 추출 방식 사용
                keywords = [word for word in text.split() 
                          if len(word) > 1]
        else:
            # 기본 키워드 추출
            keywords = [word for word in text.split() 
                       if len(word) > 1]
        
        return list(set(keywords))

    def analyze_exercise_level(self, text: str) -> str:
        """운동 난이도 분석"""
        # 빈 텍스트 처리
        if not text or not text.strip():
            return '기본'
            
        text = text.lower()
        scores = {level: 0 for level in ['기본', '중급', '고급']}
        
        for level, keywords in self.exercise_keywords.items():
            if level in ['기본', '중급', '고급']:
                for keyword in keywords:
                    if keyword in text:
                        scores[level] += 1
        
        max_score = max(scores.values())
        if max_score == 0:
            return '기본'
        
        return max(scores.items(), key=lambda x: x[1])[0]

    def analyze_exercise_focus(self, text: str) -> List[str]:
        """운동 초점 분석"""
        # 빈 텍스트 처리
        if not text or not text.strip():
            return []
            
        focus_areas = []
        
        # 신체 부위 키워드
        body_parts = {
            '코어': ['코어', '복근', '복부', '허리'],
            '상체': ['어깨', '팔', '가슴', '등'],
            '하체': ['다리', '엉덩이', '무릎', '발목'],
            '전신': ['전신', '전체', '바디']
        }
        
        for area, keywords in body_parts.items():
            for keyword in keywords:
                if keyword in text:
                    focus_areas.append(area)
                    break
        
        return list(set(focus_areas))

    def extract_exercise_components(self, text: str) -> Dict:
        """운동 구성요소 추출"""
        # 빈 텍스트 처리
        if not text or not text.strip():
            return {
                'preparation': "",
                'execution': [],
                'breathing': "",
                'precautions': []
            }
            
        components = {
            'preparation': self.extract_preparation(text),
            'execution': self.extract_execution(text),
            'breathing': self.extract_breathing(text),
            'precautions': self.extract_precautions(text)
        }
        
        return components

    def extract_preparation(self, text: str) -> str:
        """준비 자세 추출"""
        pattern = r'준비(?:자세|동작)[:\s]+(.*?)(?=\n\n|$)'
        match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
        return match.group(1).strip() if match else ""

    def extract_execution(self, text: str) -> List[str]:
        """실행 단계 추출"""
        pattern = r'(?:실행|동작)[:\s]+(.*?)(?=\n\n|$)'
        match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
        if match:
            steps = match.group(1).split('\n')
            return [step.strip() for step in steps if step.strip()]
        return []

    def extract_breathing(self, text: str) -> str:
        """호흡법 추출"""
        pattern = r'호흡[:\s]+(.*?)(?=\n\n|$)'
        match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
        return match.group(1).strip() if match else ""

    def extract_precautions(self, text: str) -> List[str]:
        """주의사항 추출"""
        pattern = r'주의(?:사항)?[:\s]+(.*?)(?=\n\n|$)'
        match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
        if match:
            precautions = match.group(1).split('\n')
            return [p.strip() for p in precautions if p.strip()]
        return []

    def analyze_difficulty(self, text: str) -> Dict:
        """운동 난이도 상세 분석"""
        # 빈 텍스트 처리
        if not text or not text.strip():
            return {
                'overall_level': '기본',
                'complexity': 5,
                'intensity': 5,
                'required_experience': '초보자'
            }
            
        difficulty_factors = {
            'overall_level': self.analyze_exercise_level(text),
            'complexity': self.analyze_complexity(text),
            'intensity': self.analyze_intensity(text),
            'required_experience': self.analyze_required_experience(text)
        }
        return difficulty_factors

    def analyze_complexity(self, text: str) -> int:
        """동작 복잡도 분석 (1-10)"""
        if not text or not text.strip():
            return 5
            
        complexity_score = 5  # 기본값
        
        # 복잡도 증가 요인
        if '고급' in text or '전문가' in text:
            complexity_score += 2
        if '중급' in text or '심화' in text:
            complexity_score += 1
        if '조합' in text or '연결' in text:
            complexity_score += 1
        
        # 복잡도 감소 요인
        if '기본' in text or '초급' in text:
            complexity_score -= 1
        if '단순' in text or '쉬운' in text:
            complexity_score -= 1
            
        return max(1, min(10, complexity_score))

    def analyze_intensity(self, text: str) -> int:
        """운동 강도 분석 (1-10)"""
        if not text or not text.strip():
            return 5
            
        intensity_score = 5  # 기본값
        
        # 강도 증가 요인
        if '고강도' in text or '힘든' in text:
            intensity_score += 2
        if '중강도' in text:
            intensity_score += 1
        if '반복' in text or '지속' in text:
            intensity_score += 1
            
        # 강도 감소 요인
        if '저강도' in text or '가벼운' in text:
            intensity_score -= 1
        if '천천히' in text or '편안' in text:
            intensity_score -= 1
            
        return max(1, min(10, intensity_score))

    def analyze_required_experience(self, text: str) -> str:
        """필요 경험 수준 분석"""
        if not text or not text.strip():
            return '초보자'
            
        if '고급' in text or '전문가' in text:
            return '상급자'
        elif '중급' in text or '심화' in text:
            return '중급자'
        else:
            return '초보자'

class KoreanProgramGenerator:
    def __init__(self, text_processor: KoreanTextProcessor):
        """한국어 프로그램 생성기 초기화"""
        self.text_processor = text_processor
        
    def generate_korean_program(self, assessment: Dict, exercises: List[Dict]) -> str:
        """맞춤형 운동 프로그램 생성"""
        if not assessment or not exercises:
            return "프로그램을 생성할 수 없습니다. 평가 데이터나 운동 데이터가 부족합니다."
            
        program = f"""
        [필라테스 맞춤 운동 프로그램]
        
        환자 정보:
        - 상태: {assessment.get('condition', '')}
        - 통증 수준: {assessment.get('pain_level', 0)}/10
        - 운동 경험: {assessment.get('experience_level', '초보자')}
        
        추천 운동:
        """
        
        for i, exercise in enumerate(exercises, 1):
            program += f"""
        {i}. {exercise.get('name', '')}
           - 난이도: {exercise.get('difficulty', '기본')}
           - 세트: {exercise.get('sets', 3)}회
           - 반복: {exercise.get('reps', 10)}회
           - 주의사항: {', '.join(exercise.get('precautions', []))}
            """
        
        program += """
        * 모든 운동은 통증이 없는 범위 내에서 수행하세요.
        * 불편함을 느끼면 즉시 중단하고 전문가와 상담하세요.
        """
        
        return program