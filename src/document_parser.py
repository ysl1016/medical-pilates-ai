"""
Document Parser Module for Medical Pilates AI System
Handles parsing and processing of various document formats
"""

import os
from typing import Dict, List, Optional, Tuple
import fitz  # PyMuPDF
from docx import Document
import pandas as pd
from PIL import Image
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import logging

class DocumentParser:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the DocumentParser with specified models and configurations
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.text_classifier = pipeline("text-classification", model="distilbert-base-uncased")
        self.logger = logging.getLogger(__name__)
        
    def parse_pdf(self, file_path: str) -> Tuple[str, List[Dict], List[Dict]]:
        """
        Parse PDF files and extract text, images, and tables
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            Tuple containing:
            - Extracted text (str)
            - List of extracted images with metadata (List[Dict])
            - List of extracted tables with metadata (List[Dict])
        """
        try:
            doc = fitz.open(file_path)
            text = ""
            images = []
            tables = []
            
            for page_num, page in enumerate(doc):
                # Extract text
                text += page.get_text()
                
                # Extract images
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_data = {
                        "page": page_num + 1,
                        "index": img_index,
                        "data": base_image["image"],
                        "metadata": base_image
                    }
                    images.append(image_data)
                
                # Extract tables (simplified)
                tables_on_page = self._extract_tables_from_page(page)
                tables.extend(tables_on_page)
            
            return text, images, tables
            
        except Exception as e:
            self.logger.error(f"Error parsing PDF {file_path}: {str(e)}")
            return "", [], []
            
    def parse_docx(self, file_path: str) -> Tuple[str, List[Dict], List[Dict]]:
        """
        Parse DOCX files and extract text, images, and tables
        
        Args:
            file_path (str): Path to the DOCX file
            
        Returns:
            Tuple containing:
            - Extracted text (str)
            - List of extracted images with metadata (List[Dict])
            - List of extracted tables with metadata (List[Dict])
        """
        try:
            doc = Document(file_path)
            text = ""
            images = []
            tables = []
            
            # Extract text
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            # Extract tables
            for table in doc.tables:
                table_data = []
                for row in table.rows:
                    row_data = [cell.text for cell in row.cells]
                    table_data.append(row_data)
                tables.append({"data": table_data})
            
            # Note: Image extraction from DOCX requires additional processing
            # This is a simplified version
            
            return text, images, tables
            
        except Exception as e:
            self.logger.error(f"Error parsing DOCX {file_path}: {str(e)}")
            return "", [], []
    
    def process_text(self, text: str) -> Dict:
        """
        Process extracted text using NLP techniques
        
        Args:
            text (str): Input text to process
            
        Returns:
            Dict containing processed text data and metadata
        """
        try:
            # Generate embeddings
            embeddings = self.model.encode(text)
            
            # Classify text
            classification = self.text_classifier(text[:512])[0]
            
            return {
                "text": text,
                "embeddings": embeddings,
                "classification": classification,
                "length": len(text)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing text: {str(e)}")
            return {"text": text, "error": str(e)}
    
    def _extract_tables_from_page(self, page) -> List[Dict]:
        """
        Helper method to extract tables from a PDF page
        
        Args:
            page: PDF page object
            
        Returns:
            List of extracted tables with metadata
        """
        # This is a simplified version
        # Real implementation would need more sophisticated table detection
        tables = []
        # Add table extraction logic here
        return tables
    
    def save_processed_data(self, output_path: str, data: Dict):
        """
        Save processed document data to disk
        
        Args:
            output_path (str): Path to save the processed data
            data (Dict): Processed document data
        """
        try:
            # Save text and metadata
            with open(os.path.join(output_path, "text.txt"), "w", encoding="utf-8") as f:
                f.write(data["text"])
            
            # Save embeddings
            if "embeddings" in data:
                np.save(os.path.join(output_path, "embeddings.npy"), data["embeddings"])
            
            # Save classification
            if "classification" in data:
                with open(os.path.join(output_path, "metadata.txt"), "w") as f:
                    f.write(str(data["classification"]))
                    
        except Exception as e:
            self.logger.error(f"Error saving processed data: {str(e)}")