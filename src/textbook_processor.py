"""
Textbook Processing Module for Medical Pilates AI
Processes and analyzes medical pilates textbook content
"""

import os
from typing import Dict, List, Optional
import pandas as pd
from document_parser import DocumentParser
from korean_utils import KoreanTextProcessor
import logging

class TextbookProcessor:
    def __init__(self, base_path: str):
        """
        Initialize the TextbookProcessor
        
        Args:
            base_path (str): Base directory for textbook files
        """
        self.base_path = base_path
        self.document_parser = DocumentParser(base_path)
        self.korean_processor = KoreanTextProcessor()
        self.logger = logging.getLogger(__name__)
        
        # Initialize exercise database
        self.exercises_db = pd.DataFrame(columns=[
            'name',
            'description',
            'difficulty_level',
            'target_areas',
            'precautions',
            'variations'
        ])
    
    def process_textbook(self, textbook_path: str) -> Dict:
        """
        Process a medical pilates textbook
        
        Args:
            textbook_path (str): Path to the textbook file
            
        Returns:
            Dict containing processed textbook content
        """
        try:
            # Parse textbook content
            content = self.document_parser.parse_file(textbook_path)
            
            # Extract and process exercises
            exercises = self._extract_exercises(content)
            
            # Process Korean text
            processed_content = {
                'title': content.get('title', ''),
                'author': content.get('author', ''),
                'publication_date': content.get('date', ''),
                'exercises': [
                    self.korean_processor.process_exercise_description(exercise)
                    for exercise in exercises
                ]
            }
            
            # Update exercise database
            self._update_exercise_database(processed_content['exercises'])
            
            return processed_content
            
        except Exception as e:
            self.logger.error(f"Error processing textbook {textbook_path}: {str(e)}")
            return {}
    
    def _extract_exercises(self, content: Dict) -> List[Dict]:
        """
        Extract exercise information from textbook content
        
        Args:
            content (Dict): Parsed textbook content
            
        Returns:
            List of exercise dictionaries
        """
        exercises = []
        
        try:
            text = content.get('text', '')
            sections = text.split('\n\n')
            
            current_exercise = {}
            for section in sections:
                if section.strip().startswith('운동명:') or section.strip().startswith('Exercise:'):
                    if current_exercise:
                        exercises.append(current_exercise.copy())
                    current_exercise = {'name': section.split(':', 1)[1].strip()}
                elif current_exercise:
                    if section.strip().startswith('설명:') or section.strip().startswith('Description:'):
                        current_exercise['description'] = section.split(':', 1)[1].strip()
                    elif section.strip().startswith('난이도:') or section.strip().startswith('Difficulty:'):
                        current_exercise['difficulty_level'] = section.split(':', 1)[1].strip()
                    elif section.strip().startswith('주의사항:') or section.strip().startswith('Precautions:'):
                        current_exercise['precautions'] = [
                            p.strip() for p in section.split(':', 1)[1].split(',')
                        ]
            
            if current_exercise:
                exercises.append(current_exercise)
                
        except Exception as e:
            self.logger.error(f"Error extracting exercises: {str(e)}")
            
        return exercises
    
    def _update_exercise_database(self, exercises: List[Dict]):
        """
        Update the exercise database with new exercises
        
        Args:
            exercises (List[Dict]): List of processed exercises
        """
        try:
            for exercise in exercises:
                # Check if exercise already exists
                existing = self.exercises_db[
                    self.exercises_db['name'] == exercise.get('name_ko', '')
                ]
                
                if existing.empty:
                    # Add new exercise
                    self.exercises_db = self.exercises_db.append({
                        'name': exercise.get('name_ko', ''),
                        'description': exercise.get('description_ko', ''),
                        'difficulty_level': exercise.get('difficulty_level_ko', '기본'),
                        'target_areas': exercise.get('target_areas_ko', []),
                        'precautions': exercise.get('precautions_ko', []),
                        'variations': exercise.get('variations_ko', [])
                    }, ignore_index=True)
                    
        except Exception as e:
            self.logger.error(f"Error updating exercise database: {str(e)}")
    
    def search_exercises(self, query: str, 
                        difficulty: Optional[str] = None,
                        target_area: Optional[str] = None) -> pd.DataFrame:
        """
        Search for exercises in the database
        
        Args:
            query (str): Search query
            difficulty (str, optional): Filter by difficulty level
            target_area (str, optional): Filter by target area
            
        Returns:
            DataFrame of matching exercises
        """
        try:
            # Start with full database
            results = self.exercises_db.copy()
            
            # Apply filters
            if query:
                # Search in name and description
                mask = (
                    results['name'].str.contains(query, case=False, na=False) |
                    results['description'].str.contains(query, case=False, na=False)
                )
                results = results[mask]
            
            if difficulty:
                results = results[results['difficulty_level'] == difficulty]
                
            if target_area:
                results = results[results['target_areas'].apply(
                    lambda x: target_area in x if isinstance(x, list) else False
                )]
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching exercises: {str(e)}")
            return pd.DataFrame()
    
    def get_exercise_details(self, exercise_name: str) -> Dict:
        """
        Get detailed information about a specific exercise
        
        Args:
            exercise_name (str): Name of the exercise
            
        Returns:
            Dict containing exercise details
        """
        try:
            exercise = self.exercises_db[
                self.exercises_db['name'] == exercise_name
            ].iloc[0].to_dict()
            
            # Generate exercise summary
            exercise['summary'] = self.korean_processor.create_exercise_summary({
                'name_ko': exercise['name'],
                'description_ko': exercise['description'],
                'precautions_ko': exercise['precautions'],
                'difficulty_level_ko': exercise['difficulty_level']
            })
            
            return exercise
            
        except Exception as e:
            self.logger.error(f"Error getting exercise details: {str(e)}")
            return {}