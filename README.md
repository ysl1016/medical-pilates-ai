# Medical Pilates AI Prescription System

An advanced AI-powered system for personalized Pilates exercise prescriptions based on patient assessments and medical documentation.

## Features

### 1. Document Processing
- Parse multiple document formats (PDF, DOCX, TXT, MD)
- Extract images and tables from documents
- Manage document metadata and relationships
- Organize medical literature and patient records

### 2. Media Management
- Process image and video files
- Create thumbnails and galleries
- Organize media assets by categories
- Support for exercise demonstration videos

### 3. RAG (Retrieval-Augmented Generation) System
- Document embedding and semantic search
- Context-based response generation
- Intelligent exercise recommendation
- Personalized prescription generation

### 4. Patient Assessment
- Comprehensive condition evaluation
- Movement pattern analysis
- Pain point identification
- Progress tracking

### 5. Exercise Prescription
- Personalized exercise selection
- Difficulty level adjustment
- Progression planning
- Contraindication checking

### 6. Web Interface
- User-friendly Gradio interface
- Real-time assessment input
- Instant prescription generation
- Exercise visualization

## Setup and Installation

1. Mount Google Drive
2. Install required libraries
3. Set up project directory structure
4. Initialize utility classes
5. Process document database
6. Launch Gradio interface

## Dependencies

- transformers
- sentence-transformers
- faiss-cpu
- torch
- pandas
- numpy
- tqdm
- gradio
- python-docx
- Pillow
- pdf2image

## Usage

1. Upload medical documents and exercise media
2. Process and index the document database
3. Input patient assessment data
4. Generate personalized exercise prescriptions
5. Review and modify recommendations
6. Export final prescription