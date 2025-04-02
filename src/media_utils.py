import cv2
import numpy as np
from PIL import Image
import os
from typing import List, Tuple, Optional, Union
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

class MediaProcessor:
    def __init__(self):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        self.model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
        
    def load_image(self, image_path: str) -> Image.Image:
        """이미지 파일을 로드합니다."""
        return Image.open(image_path)
    
    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """이미지를 전처리하여 모델 입력에 적합한 형태로 변환합니다."""
        if isinstance(image, str):
            image = self.load_image(image)
        
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        return inputs.pixel_values
    
    def analyze_posture(self, image_path: str) -> dict:
        """이미지에서 자세를 분석합니다."""
        image = self.load_image(image_path)
        inputs = self.preprocess_image(image)
        
        with torch.no_grad():
            outputs = self.model(inputs)
            predictions = outputs.logits.softmax(dim=-1)
        
        return {
            "posture_quality": float(predictions[0][0]),  # 자세의 정확도
            "potential_issues": self._identify_posture_issues(predictions)
        }
    
    def _identify_posture_issues(self, predictions: torch.Tensor) -> List[str]:
        """자세 분석 결과에서 잠재적인 문제점을 식별합니다."""
        issues = []
        threshold = 0.3
        
        # 예시 이슈 카테고리
        issue_categories = [
            "척추 정렬",
            "골반 정렬",
            "어깨 정렬",
            "무릎 정렬"
        ]
        
        for i, prob in enumerate(predictions[0]):
            if prob > threshold and i < len(issue_categories):
                issues.append(issue_categories[i])
        
        return issues
    
    def extract_video_frames(self, video_path: str, interval: int = 30) -> List[np.ndarray]:
        """비디오에서 프레임을 추출합니다."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if len(frames) % interval == 0:
                frames.append(frame)
                
        cap.release()
        return frames
    
    def analyze_movement(self, video_path: str) -> dict:
        """운동 동작을 분석합니다."""
        frames = self.extract_video_frames(video_path)
        frame_analyses = []
        
        for frame in frames:
            # PIL Image로 변환
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = self.preprocess_image(pil_image)
            
            with torch.no_grad():
                outputs = self.model(inputs)
                predictions = outputs.logits.softmax(dim=-1)
                
            frame_analyses.append({
                "posture_quality": float(predictions[0][0]),
                "issues": self._identify_posture_issues(predictions)
            })
        
        return {
            "frame_count": len(frames),
            "average_quality": sum(f["posture_quality"] for f in frame_analyses) / len(frame_analyses),
            "frame_analyses": frame_analyses
        }
    
    def save_analysis_visualization(self, image_path: str, analysis_result: dict, output_path: str):
        """분석 결과를 시각화하여 저장합니다."""
        image = cv2.imread(image_path)
        
        # 분석 결과 텍스트 추가
        text_y = 30
        cv2.putText(image, f"Posture Quality: {analysis_result['posture_quality']:.2f}", 
                   (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        for issue in analysis_result.get('potential_issues', []):
            text_y += 30
            cv2.putText(image, f"Issue: {issue}", 
                       (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imwrite(output_path, image)
        
    def create_progress_visualization(self, image_paths: List[str], analyses: List[dict], output_path: str):
        """여러 이미지의 진행 상황을 시각화합니다."""
        num_images = len(image_paths)
        if num_images == 0:
            return
            
        # 이미지 그리드 생성
        grid_size = int(np.ceil(np.sqrt(num_images)))
        grid = np.zeros((grid_size * 300, grid_size * 300, 3), dtype=np.uint8)
        
        for i, (img_path, analysis) in enumerate(zip(image_paths, analyses)):
            row = i // grid_size
            col = i % grid_size
            
            img = cv2.imread(img_path)
            img = cv2.resize(img, (300, 300))
            
            # 분석 결과 텍스트 추가
            cv2.putText(img, f"Quality: {analysis['posture_quality']:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            grid[row*300:(row+1)*300, col*300:(col+1)*300] = img
            
        cv2.imwrite(output_path, grid)