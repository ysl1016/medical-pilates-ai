"""
Medical Pilates AI System Configuration
의료 필라테스 AI 시스템 설정
"""

import os
from pathlib import Path
from typing import Dict, Any

class SystemConfig:
    """시스템 기본 설정"""
    
    # 프로젝트 기본 경로
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # 데이터 저장 경로
    DATA_DIR = BASE_DIR / "data"
    EXERCISE_DB_PATH = DATA_DIR / "exercises.json"
    MEDIA_DIR = DATA_DIR / "media"
    LOG_DIR = BASE_DIR / "logs"
    
    # 임베딩 설정
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIMENSION = 384
    
    # RAG 시스템 설정
    MAX_CHUNKS = 5
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    
    # 언어 모델 설정
    LLM_MODEL = "gpt-3.5-turbo"
    MAX_TOKENS = 2000
    TEMPERATURE = 0.7
    
    # 번역 설정
    TRANSLATION_BATCH_SIZE = 32
    MAX_TEXT_LENGTH = 512
    
    # 미디어 처리 설정
    IMAGE_MAX_SIZE = (800, 800)
    THUMBNAIL_SIZE = (200, 200)
    SUPPORTED_IMAGE_TYPES = [".jpg", ".jpeg", ".png", ".gif"]
    SUPPORTED_VIDEO_TYPES = [".mp4", ".avi", ".mov"]
    
    # 캐시 설정
    CACHE_DIR = BASE_DIR / "cache"
    CACHE_EXPIRY = 3600  # 1시간
    
    @classmethod
    def initialize(cls) -> None:
        """필요한 디렉토리 생성"""
        dirs = [
            cls.DATA_DIR,
            cls.MEDIA_DIR,
            cls.LOG_DIR,
            cls.CACHE_DIR
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """현재 설정값 반환"""
        return {
            "base_dir": str(cls.BASE_DIR),
            "data_dir": str(cls.DATA_DIR),
            "embedding_model": cls.EMBEDDING_MODEL,
            "embedding_dimension": cls.EMBEDDING_DIMENSION,
            "llm_model": cls.LLM_MODEL,
            "max_tokens": cls.MAX_TOKENS,
            "temperature": cls.TEMPERATURE
        }

class DevelopmentConfig(SystemConfig):
    """개발 환경 설정"""
    DEBUG = True
    TESTING = False
    
    # 로깅 설정
    LOG_LEVEL = "DEBUG"
    LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
    
    # 개발용 API 키 (환경 변수에서 로드)
    API_KEYS = {
        "openai": os.getenv("OPENAI_API_KEY", ""),
        "google_cloud": os.getenv("GOOGLE_CLOUD_API_KEY", "")
    }

class ProductionConfig(SystemConfig):
    """운영 환경 설정"""
    DEBUG = False
    TESTING = False
    
    # 로깅 설정
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    # 성능 최적화 설정
    CACHE_EXPIRY = 86400  # 24시간
    EMBEDDING_BATCH_SIZE = 64
    
    # 보안 설정
    REQUIRE_AUTH = True
    SESSION_LIFETIME = 3600
    
    @classmethod
    def validate_api_keys(cls) -> bool:
        """API 키 유효성 검증"""
        required_keys = ["openai", "google_cloud"]
        return all(
            os.getenv(f"{key.upper()}_API_KEY")
            for key in required_keys
        )

class TestingConfig(SystemConfig):
    """테스트 환경 설정"""
    DEBUG = True
    TESTING = True
    
    # 테스트용 데이터 경로
    TEST_DATA_DIR = SystemConfig.BASE_DIR / "tests" / "data"
    
    # 테스트 설정
    USE_MOCK_API = True
    SKIP_HEAVY_TESTS = False
    
    @classmethod
    def setup_test_env(cls) -> None:
        """테스트 환경 설정"""
        cls.initialize()
        cls.TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # 테스트용 더미 데이터 생성
        test_exercise_db = cls.TEST_DATA_DIR / "test_exercises.json"
        if not test_exercise_db.exists():
            with open(test_exercise_db, "w") as f:
                f.write("{}")

# 환경별 설정 매핑
config_by_env = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig
}

def get_config(env: str = None) -> SystemConfig:
    """환경에 따른 설정 클래스 반환"""
    if env is None:
        env = os.getenv("APP_ENV", "development")
    
    config_class = config_by_env.get(env.lower(), DevelopmentConfig)
    return config_class