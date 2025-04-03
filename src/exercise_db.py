"""
Exercise Database Module for Medical Pilates AI
운동 데이터베이스 모듈 - 필라테스 운동 정보 관리
"""

from typing import Dict, List, Optional, Iterator
import json
import os
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import ijson  # JSON 스트리밍 파싱을 위한 라이브러리

@dataclass
class Exercise:
    """운동 정보를 저장하는 데이터 클래스"""
    id: str
    name_ko: str
    name_en: str
    description_ko: str
    description_en: str
    difficulty_level: str
    target_areas: List[str]
    contraindications: List[str]
    prerequisites: List[str]
    equipment_needed: List[str]
    steps_ko: List[str]
    steps_en: List[str]
    tips_ko: List[str]
    tips_en: List[str]
    variations: List[Dict]
    created_at: str = datetime.now().isoformat()
    updated_at: str = datetime.now().isoformat()

class ExerciseDatabase:
    """필라테스 운동 데이터베이스 관리 클래스 - 메모리 효율성 개선"""
    
    def __init__(self, db_path: str = "data/exercises.json"):
        """
        데이터베이스 초기화
        
        Args:
            db_path: JSON 데이터베이스 파일 경로
        """
        self.db_path = db_path
        self.exercises: Dict[str, Exercise] = {}
        self.logger = logging.getLogger(__name__)
        
        # 로거 설정이 되어있지 않은 경우
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[logging.StreamHandler()]
            )
        
        self._load_database()
        
    def _load_database(self) -> None:
        """데이터베이스 파일 로드 - 메모리 효율성 개선"""
        if os.path.exists(self.db_path):
            try:
                # 대용량 데이터베이스를 위한 스트리밍 방식 로드
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    # JSON 객체 전체를 로드하는 대신 각 키-값 쌍을 개별적으로 처리
                    exercises_loaded = 0
                    
                    # JSON 객체로부터 키-값 쌍 스트리밍
                    for exercise_id, exercise_data in ijson.kvitems(f, ''):
                        try:
                            exercise = Exercise(**exercise_data)
                            self.exercises[exercise_id] = exercise
                            exercises_loaded += 1
                        except Exception as e:
                            self.logger.error(f"운동 데이터 로드 중 오류: ID {exercise_id}, {str(e)}")
                    
                    self.logger.info(f"{exercises_loaded}개의 운동 데이터 로드 완료")
            except Exception as e:
                self.logger.error(f"데이터베이스 로드 중 오류 발생: {str(e)}")
                self._initialize_default_exercises()
        else:
            self.logger.info("데이터베이스 파일이 없어 기본 운동 데이터 초기화")
            self._initialize_default_exercises()
    
    def _save_database(self) -> None:
        """데이터베이스를 파일에 저장 - 메모리 효율성 개선"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        try:
            # 전체 데이터를 일괄로 저장하는 대신, 한 번에 한 항목씩 저장
            with open(self.db_path, 'w', encoding='utf-8') as f:
                # JSON 객체 시작
                f.write("{\n")
                
                # 각 운동을 개별적으로 직렬화하고 저장
                entries = list(self.exercises.items())
                for i, (ex_id, ex) in enumerate(entries):
                    # 각 항목을 개별적으로 직렬화
                    ex_json = json.dumps(asdict(ex), ensure_ascii=False)
                    
                    # JSON 형식으로 키와 값 쌍 작성
                    f.write(f'  "{ex_id}": {ex_json}')
                    
                    # 마지막 항목이 아니면 쉼표 추가
                    if i < len(entries) - 1:
                        f.write(",\n")
                    else:
                        f.write("\n")
                
                # JSON 객체 종료
                f.write("}")
                
            self.logger.info(f"{len(self.exercises)}개의 운동 데이터 저장 완료")
        except Exception as e:
            self.logger.error(f"데이터베이스 저장 중 오류 발생: {str(e)}")
    
    def _initialize_default_exercises(self) -> None:
        """기본 운동 데이터 초기화"""
        default_exercises = [
            Exercise(
                id="basic_breathing",
                name_ko="기본 호흡법",
                name_en="Basic Breathing",
                description_ko="필라테스의 기본이 되는 호흡법 훈련",
                description_en="Basic breathing technique training for Pilates",
                difficulty_level="beginner",
                target_areas=["코어", "호흡근"],
                contraindications=["심각한 호흡기 질환"],
                prerequisites=[],
                equipment_needed=[],
                steps_ko=[
                    "바로 누운 자세로 시작",
                    "복부에 손을 올리고 호흡 관찰",
                    "들숨시 복부 팽창",
                    "날숨시 복부 수축"
                ],
                steps_en=[
                    "Start in supine position",
                    "Place hands on abdomen and observe breathing",
                    "Expand abdomen on inhale",
                    "Contract abdomen on exhale"
                ],
                tips_ko=[
                    "자연스러운 호흡 유지",
                    "과도한 힘 주지 않기"
                ],
                tips_en=[
                    "Maintain natural breathing",
                    "Avoid excessive force"
                ],
                variations=[]
            ),
            Exercise(
                id="hundred_prep",
                name_ko="헌드레드 준비 운동",
                name_en="Hundred Prep",
                description_ko="코어 강화를 위한 기초 운동",
                description_en="Basic exercise for core strengthening",
                difficulty_level="beginner",
                target_areas=["코어", "복근"],
                contraindications=["급성 요통", "임신 후기"],
                prerequisites=["기본 호흡법"],
                equipment_needed=["매트"],
                steps_ko=[
                    "바로 누운 자세에서 무릎 구부리기",
                    "골반 중립 자세 유지",
                    "턱을 가슴쪽으로 당기며 상체 들기",
                    "팔을 몸통 옆에 두고 펌핑"
                ],
                steps_en=[
                    "Lie supine with knees bent",
                    "Maintain neutral pelvis",
                    "Curl upper body with chin tucked",
                    "Pump arms beside torso"
                ],
                tips_ko=[
                    "복부 긴장 유지",
                    "목 긴장 최소화"
                ],
                tips_en=[
                    "Maintain abdominal engagement",
                    "Minimize neck tension"
                ],
                variations=[
                    {
                        "name_ko": "다리 뻗기",
                        "name_en": "Leg extension",
                        "description_ko": "다리를 45도로 뻗어 난이도 증가",
                        "description_en": "Increase difficulty by extending legs to 45 degrees"
                    }
                ]
            )
        ]
        
        for exercise in default_exercises:
            self.exercises[exercise.id] = exercise
        self._save_database()
    
    def add_exercise(self, exercise: Exercise) -> bool:
        """
        새로운 운동 추가
        
        Args:
            exercise: 추가할 운동 객체
            
        Returns:
            bool: 성공 여부
        """
        try:
            if exercise.id in self.exercises:
                self.logger.warning(f"이미 존재하는 운동 ID입니다: {exercise.id}")
                return False
            
            self.exercises[exercise.id] = exercise
            self._save_database()
            return True
        except Exception as e:
            self.logger.error(f"운동 추가 중 오류 발생: {str(e)}")
            return False
    
    def get_exercise(self, exercise_id: str) -> Optional[Exercise]:
        """
        ID로 운동 정보 조회
        
        Args:
            exercise_id: 조회할 운동 ID
            
        Returns:
            Exercise 객체 또는 None
        """
        return self.exercises.get(exercise_id)
    
    def update_exercise(self, exercise_id: str, updated_data: Dict) -> bool:
        """
        운동 정보 업데이트
        
        Args:
            exercise_id: 업데이트할 운동 ID
            updated_data: 업데이트할 데이터
            
        Returns:
            bool: 성공 여부
        """
        try:
            if exercise_id not in self.exercises:
                self.logger.warning(f"존재하지 않는 운동 ID입니다: {exercise_id}")
                return False
            
            exercise = self.exercises[exercise_id]
            for key, value in updated_data.items():
                if hasattr(exercise, key):
                    setattr(exercise, key, value)
            
            exercise.updated_at = datetime.now().isoformat()
            self._save_database()
            return True
        except Exception as e:
            self.logger.error(f"운동 업데이트 중 오류 발생: {str(e)}")
            return False
    
    def delete_exercise(self, exercise_id: str) -> bool:
        """
        운동 삭제
        
        Args:
            exercise_id: 삭제할 운동 ID
            
        Returns:
            bool: 성공 여부
        """
        try:
            if exercise_id not in self.exercises:
                self.logger.warning(f"존재하지 않는 운동 ID입니다: {exercise_id}")
                return False
            
            del self.exercises[exercise_id]
            self._save_database()
            return True
        except Exception as e:
            self.logger.error(f"운동 삭제 중 오류 발생: {str(e)}")
            return False
    
    def search_exercises(self, 
                        difficulty: Optional[str] = None,
                        target_area: Optional[str] = None,
                        equipment: Optional[str] = None) -> List[Exercise]:
        """
        조건에 맞는 운동 검색
        
        Args:
            difficulty: 난이도
            target_area: 목표 부위
            equipment: 필요 장비
            
        Returns:
            검색 조건에 맞는 Exercise 객체 리스트
        """
        try:
            results = []
            
            for exercise in self.exercises.values():
                matches = True
                
                if difficulty and exercise.difficulty_level != difficulty:
                    matches = False
                if target_area and target_area not in exercise.target_areas:
                    matches = False
                if equipment and equipment not in exercise.equipment_needed:
                    matches = False
                    
                if matches:
                    results.append(exercise)
                    
            return results
        except Exception as e:
            self.logger.error(f"운동 검색 중 오류 발생: {str(e)}")
            return []
    
    def get_exercise_progression(self, exercise_id: str) -> List[Exercise]:
        """
        특정 운동의 난이도별 진행 경로 제공
        
        Args:
            exercise_id: 기준 운동 ID
            
        Returns:
            난이도 순으로 정렬된 Exercise 객체 리스트
        """
        try:
            base_exercise = self.get_exercise(exercise_id)
            if not base_exercise:
                self.logger.warning(f"존재하지 않는 운동 ID입니다: {exercise_id}")
                return []
            
            # 같은 목표 부위를 가진 운동들을 난이도별로 정렬
            related_exercises = [
                ex for ex in self.exercises.values()
                if any(area in base_exercise.target_areas for area in ex.target_areas)
            ]
            
            difficulty_order = {
                "beginner": 0,
                "intermediate": 1,
                "advanced": 2
            }
            
            return sorted(
                related_exercises,
                key=lambda x: difficulty_order.get(x.difficulty_level, 999)
            )
        except Exception as e:
            self.logger.error(f"운동 진행 경로 검색 중 오류 발생: {str(e)}")
            return []
    
    def get_complementary_exercises(self, exercise_id: str) -> List[Exercise]:
        """
        보완 운동 추천
        
        Args:
            exercise_id: 기준 운동 ID
            
        Returns:
            추천된 보완 운동 리스트
        """
        try:
            base_exercise = self.get_exercise(exercise_id)
            if not base_exercise:
                self.logger.warning(f"존재하지 않는 운동 ID입니다: {exercise_id}")
                return []
            
            # 현재 운동과 다른 부위를 타겟팅하는 운동 찾기
            other_areas = set()
            for ex in self.exercises.values():
                other_areas.update(ex.target_areas)
            
            target_areas = set(base_exercise.target_areas)
            complementary_areas = other_areas - target_areas
            
            return [
                ex for ex in self.exercises.values()
                if any(area in complementary_areas for area in ex.target_areas)
                and ex.difficulty_level == base_exercise.difficulty_level
            ]
        except Exception as e:
            self.logger.error(f"보완 운동 추천 중 오류 발생: {str(e)}")
            return []
    
    def export_to_json(self, filepath: str) -> bool:
        """
        데이터베이스를 JSON 파일로 내보내기
        
        Args:
            filepath: 저장할 파일 경로
            
        Returns:
            bool: 성공 여부
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                data = {ex_id: asdict(ex) for ex_id, ex in self.exercises.items()}
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"데이터 내보내기 중 오류 발생: {str(e)}")
            return False
    
    def import_from_json(self, filepath: str) -> bool:
        """
        JSON 파일에서 데이터베이스 가져오기
        
        Args:
            filepath: 가져올 파일 경로
            
        Returns:
            bool: 성공 여부
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 기존 데이터 백업
                backup_exercises = self.exercises.copy()
                
                try:
                    # 새 데이터 로드
                    self.exercises = {}
                    for exercise_id, exercise_data in data.items():
                        exercise = Exercise(**exercise_data)
                        self.exercises[exercise_id] = exercise
                    
                    # 데이터베이스 저장
                    self._save_database()
                    return True
                except Exception as e:
                    # 오류 발생 시 백업에서 복원
                    self.exercises = backup_exercises
                    raise e
                
        except FileNotFoundError:
            self.logger.error(f"파일을 찾을 수 없습니다: {filepath}")
            return False
        except json.JSONDecodeError:
            self.logger.error(f"잘못된 JSON 형식입니다: {filepath}")
            return False
        except Exception as e:
            self.logger.error(f"데이터 가져오기 중 오류 발생: {str(e)}")
            return False

    def get_exercise_statistics(self) -> Dict:
        """
        데이터베이스 통계 정보 제공
        
        Returns:
            통계 정보를 담은 딕셔너리
        """
        try:
            stats = {
                "total_exercises": len(self.exercises),
                "difficulty_distribution": {
                    "beginner": 0,
                    "intermediate": 0,
                    "advanced": 0
                },
                "target_areas": {},
                "equipment_needed": {}
            }
            
            for exercise in self.exercises.values():
                # 난이도 분포
                stats["difficulty_distribution"][exercise.difficulty_level] = (
                    stats["difficulty_distribution"].get(exercise.difficulty_level, 0) + 1
                )
                
                # 목표 부위 분포
                for area in exercise.target_areas:
                    stats["target_areas"][area] = stats["target_areas"].get(area, 0) + 1
                
                # 필요 장비 분포
                for equipment in exercise.equipment_needed:
                    stats["equipment_needed"][equipment] = (
                        stats["equipment_needed"].get(equipment, 0) + 1
                    )
            
            return stats
        except Exception as e:
            self.logger.error(f"통계 정보 생성 중 오류 발생: {str(e)}")
            return {
                "total_exercises": 0,
                "error": str(e)
            }
    
    def get_exercises_lazy(self) -> Iterator[Exercise]:
        """
        메모리 효율적인 운동 데이터 조회를 위한 제너레이터
        
        Yields:
            Exercise: 데이터베이스의 각 운동 객체
        """
        for exercise in self.exercises.values():
            yield exercise