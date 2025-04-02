"""
Progress Tracker Module for Medical Pilates AI System
Tracks and analyzes patient progress over time
"""

import os
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt
import logging

class ProgressTracker:
    def __init__(self, data_dir: str):
        """
        Initialize the ProgressTracker with specified data directory
        
        Args:
            data_dir (str): Directory for storing progress data
        """
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize progress database
        self.progress_db_path = os.path.join(data_dir, "progress_db.csv")
        if not os.path.exists(self.progress_db_path):
            self._create_empty_database()
    
    def _create_empty_database(self):
        """Create an empty progress database with required columns"""
        columns = [
            "patient_id",
            "assessment_date",
            "pain_level",
            "mobility_score",
            "strength_score",
            "balance_score",
            "exercises_completed",
            "notes"
        ]
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.progress_db_path, index=False)
    
    def add_assessment(self, patient_id: str, assessment_data: Dict) -> bool:
        """
        Add a new assessment record for a patient
        
        Args:
            patient_id (str): Unique identifier for the patient
            assessment_data (Dict): Assessment data including scores and notes
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Prepare assessment record
            record = {
                "patient_id": patient_id,
                "assessment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "pain_level": assessment_data.get("pain_level", 0),
                "mobility_score": assessment_data.get("mobility_score", 0),
                "strength_score": assessment_data.get("strength_score", 0),
                "balance_score": assessment_data.get("balance_score", 0),
                "exercises_completed": assessment_data.get("exercises_completed", 0),
                "notes": assessment_data.get("notes", "")
            }
            
            # Load existing database
            df = pd.read_csv(self.progress_db_path)
            
            # Append new record
            df = df.append(record, ignore_index=True)
            
            # Save updated database
            df.to_csv(self.progress_db_path, index=False)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding assessment for patient {patient_id}: {str(e)}")
            return False
    
    def get_patient_progress(self, patient_id: str, 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> Dict:
        """
        Get progress data for a specific patient within a date range
        
        Args:
            patient_id (str): Patient identifier
            start_date (str, optional): Start date for progress data
            end_date (str, optional): End date for progress data
            
        Returns:
            Dict containing progress data and analysis
        """
        try:
            # Load database
            df = pd.read_csv(self.progress_db_path)
            
            # Filter by patient ID
            patient_data = df[df["patient_id"] == patient_id].copy()
            
            # Apply date filters if provided
            if start_date:
                patient_data = patient_data[patient_data["assessment_date"] >= start_date]
            if end_date:
                patient_data = patient_data[patient_data["assessment_date"] <= end_date]
            
            # Calculate progress metrics
            progress = {
                "assessments_count": len(patient_data),
                "latest_scores": {},
                "improvement": {},
                "trend": {}
            }
            
            if not patient_data.empty:
                # Get latest scores
                latest = patient_data.iloc[-1]
                progress["latest_scores"] = {
                    "pain_level": latest["pain_level"],
                    "mobility_score": latest["mobility_score"],
                    "strength_score": latest["strength_score"],
                    "balance_score": latest["balance_score"]
                }
                
                # Calculate improvements
                if len(patient_data) > 1:
                    first = patient_data.iloc[0]
                    for metric in ["pain_level", "mobility_score", "strength_score", "balance_score"]:
                        change = latest[metric] - first[metric]
                        progress["improvement"][metric] = change
                        
                    # Calculate trends
                    for metric in ["pain_level", "mobility_score", "strength_score", "balance_score"]:
                        values = patient_data[metric].values
                        trend = np.polyfit(range(len(values)), values, 1)[0]
                        progress["trend"][metric] = trend
            
            return progress
            
        except Exception as e:
            self.logger.error(f"Error getting progress for patient {patient_id}: {str(e)}")
            return {}
    
    def generate_progress_report(self, patient_id: str, output_path: str) -> bool:
        """
        Generate a detailed progress report for a patient
        
        Args:
            patient_id (str): Patient identifier
            output_path (str): Path to save the report
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get patient progress data
            progress = self.get_patient_progress(patient_id)
            
            if not progress:
                return False
            
            # Create visualizations
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"Patient Progress Report - {patient_id}")
            
            metrics = ["pain_level", "mobility_score", "strength_score", "balance_score"]
            
            # Load data
            df = pd.read_csv(self.progress_db_path)
            patient_data = df[df["patient_id"] == patient_id]
            
            for idx, metric in enumerate(metrics):
                ax = axes[idx // 2, idx % 2]
                values = patient_data[metric].values
                dates = pd.to_datetime(patient_data["assessment_date"])
                
                ax.plot(dates, values, 'o-')
                ax.set_title(metric.replace("_", " ").title())
                ax.set_xlabel("Date")
                ax.set_ylabel("Score")
                ax.grid(True)
                
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"{patient_id}_progress.png"))
            
            # Generate report text
            report = {
                "patient_id": patient_id,
                "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "progress_data": progress,
                "recommendations": self._generate_recommendations(progress)
            }
            
            # Save report
            with open(os.path.join(output_path, f"{patient_id}_report.json"), "w") as f:
                json.dump(report, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating report for patient {patient_id}: {str(e)}")
            return False
    
    def _generate_recommendations(self, progress: Dict) -> List[str]:
        """
        Generate exercise recommendations based on progress data
        
        Args:
            progress (Dict): Patient progress data
            
        Returns:
            List[str]: List of recommendations
        """
        recommendations = []
        
        if "latest_scores" in progress:
            scores = progress["latest_scores"]
            
            # Pain level recommendations
            if scores["pain_level"] > 5:
                recommendations.append("Consider reducing exercise intensity")
            
            # Mobility recommendations
            if scores["mobility_score"] < 7:
                recommendations.append("Focus on mobility exercises")
            
            # Strength recommendations
            if scores["strength_score"] < 7:
                recommendations.append("Incorporate more strength training")
            
            # Balance recommendations
            if scores["balance_score"] < 7:
                recommendations.append("Add balance exercises to routine")
        
        return recommendations