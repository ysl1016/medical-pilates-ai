"""
Report Generator Module for Medical Pilates AI System
Generates detailed reports and exercise prescriptions
"""

import os
from typing import Dict, List, Optional
from datetime import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Template
import logging

class ReportGenerator:
    def __init__(self, template_dir: str, output_dir: str):
        """
        Initialize the ReportGenerator with specified directories
        
        Args:
            template_dir (str): Directory containing report templates
            output_dir (str): Directory for saving generated reports
        """
        self.template_dir = template_dir
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load report templates
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Template]:
        """
        Load Jinja2 templates for different report types
        
        Returns:
            Dict[str, Template]: Dictionary of loaded templates
        """
        templates = {}
        template_files = {
            "assessment": "assessment_template.html",
            "progress": "progress_template.html",
            "prescription": "prescription_template.html"
        }
        
        try:
            for key, filename in template_files.items():
                path = os.path.join(self.template_dir, filename)
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        templates[key] = Template(f.read())
                else:
                    templates[key] = self._get_default_template(key)
                    
            return templates
            
        except Exception as e:
            self.logger.error(f"Error loading templates: {str(e)}")
            return {}
    
    def _get_default_template(self, template_type: str) -> Template:
        """
        Get default template string for different report types
        
        Args:
            template_type (str): Type of template to get
            
        Returns:
            Template: Jinja2 template object
        """
        if template_type == "assessment":
            return Template('''
                <html>
                <head>
                    <title>Patient Assessment Report</title>
                    <style>
                        body { font-family: Arial, sans-serif; }
                        .header { text-align: center; }
                        .section { margin: 20px 0; }
                        .score { font-weight: bold; }
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>Patient Assessment Report</h1>
                        <p>Date: {{ date }}</p>
                        <p>Patient ID: {{ patient_id }}</p>
                    </div>
                    
                    <div class="section">
                        <h2>Assessment Scores</h2>
                        <p>Pain Level: <span class="score">{{ scores.pain_level }}/10</span></p>
                        <p>Mobility: <span class="score">{{ scores.mobility }}/10</span></p>
                        <p>Strength: <span class="score">{{ scores.strength }}/10</span></p>
                        <p>Balance: <span class="score">{{ scores.balance }}/10</span></p>
                    </div>
                    
                    <div class="section">
                        <h2>Notes</h2>
                        <p>{{ notes }}</p>
                    </div>
                </body>
                </html>
            ''')
        elif template_type == "prescription":
            return Template('''
                <html>
                <head>
                    <title>Exercise Prescription</title>
                    <style>
                        body { font-family: Arial, sans-serif; }
                        .header { text-align: center; }
                        .exercise { margin: 20px 0; padding: 10px; border: 1px solid #ccc; }
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>Pilates Exercise Prescription</h1>
                        <p>Date: {{ date }}</p>
                        <p>Patient ID: {{ patient_id }}</p>
                    </div>
                    
                    <div class="exercises">
                        {% for exercise in exercises %}
                        <div class="exercise">
                            <h3>{{ exercise.name }}</h3>
                            <p><strong>Sets:</strong> {{ exercise.sets }}</p>
                            <p><strong>Repetitions:</strong> {{ exercise.reps }}</p>
                            <p><strong>Notes:</strong> {{ exercise.notes }}</p>
                        </div>
                        {% endfor %}
                    </div>
                    
                    <div class="section">
                        <h2>Additional Notes</h2>
                        <p>{{ notes }}</p>
                    </div>
                </body>
                </html>
            ''')
        else:
            return Template('''
                <html>
                <head>
                    <title>Medical Pilates Report</title>
                </head>
                <body>
                    <h1>{{ title }}</h1>
                    <p>{{ content }}</p>
                </body>
                </html>
            ''')
    
    def generate_assessment_report(self, patient_id: str, 
                                 assessment_data: Dict) -> str:
        """
        Generate an assessment report for a patient
        
        Args:
            patient_id (str): Patient identifier
            assessment_data (Dict): Assessment data and scores
            
        Returns:
            str: Path to the generated report file
        """
        try:
            # Prepare report data
            report_data = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "patient_id": patient_id,
                "scores": {
                    "pain_level": assessment_data.get("pain_level", 0),
                    "mobility": assessment_data.get("mobility_score", 0),
                    "strength": assessment_data.get("strength_score", 0),
                    "balance": assessment_data.get("balance_score", 0)
                },
                "notes": assessment_data.get("notes", "")
            }
            
            # Generate report using template
            report_html = self.templates["assessment"].render(**report_data)
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(
                self.output_dir, 
                f"assessment_{patient_id}_{timestamp}.html"
            )
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_html)
            
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error generating assessment report: {str(e)}")
            return ""
    
    def generate_prescription(self, patient_id: str, 
                            exercises: List[Dict],
                            notes: str = "") -> str:
        """
        Generate an exercise prescription report
        
        Args:
            patient_id (str): Patient identifier
            exercises (List[Dict]): List of prescribed exercises
            notes (str): Additional notes
            
        Returns:
            str: Path to the generated prescription file
        """
        try:
            # Prepare prescription data
            prescription_data = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "patient_id": patient_id,
                "exercises": exercises,
                "notes": notes
            }
            
            # Generate prescription using template
            prescription_html = self.templates["prescription"].render(**prescription_data)
            
            # Save prescription
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prescription_path = os.path.join(
                self.output_dir, 
                f"prescription_{patient_id}_{timestamp}.html"
            )
            
            with open(prescription_path, 'w', encoding='utf-8') as f:
                f.write(prescription_html)
            
            return prescription_path
            
        except Exception as e:
            self.logger.error(f"Error generating prescription: {str(e)}")
            return ""
    
    def generate_progress_charts(self, progress_data: Dict, 
                               output_path: str) -> bool:
        """
        Generate charts visualizing patient progress
        
        Args:
            progress_data (Dict): Patient progress data
            output_path (str): Path to save the charts
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle("Patient Progress Overview")
            
            metrics = [
                ("pain_level", "Pain Level"),
                ("mobility_score", "Mobility Score"),
                ("strength_score", "Strength Score"),
                ("balance_score", "Balance Score")
            ]
            
            for idx, (metric, title) in enumerate(metrics):
                ax = axes[idx // 2, idx % 2]
                
                if metric in progress_data:
                    dates = progress_data["dates"]
                    values = progress_data[metric]
                    
                    ax.plot(dates, values, 'o-')
                    ax.set_title(title)
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Score")
                    ax.grid(True)
                    
                    # Rotate x-axis labels for better readability
                    plt.setp(ax.get_xticklabels(), rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating progress charts: {str(e)}")
            return False
    
    def export_to_pdf(self, html_path: str, pdf_path: str) -> bool:
        """
        Convert HTML report to PDF format
        
        Args:
            html_path (str): Path to HTML report
            pdf_path (str): Path to save PDF file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # This is a placeholder for PDF conversion
            # You would typically use a library like weasyprint or pdfkit here
            self.logger.warning("PDF conversion not implemented")
            return False
            
        except Exception as e:
            self.logger.error(f"Error converting to PDF: {str(e)}")
            return False