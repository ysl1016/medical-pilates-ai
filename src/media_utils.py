"""
Media Utilities Module for Medical Pilates AI System
Handles processing and management of image and video files
"""

import os
from typing import Dict, List, Optional, Tuple
from PIL import Image
import cv2
import numpy as np
from datetime import datetime
import logging

class MediaManager:
    def __init__(self, output_dir: str, thumbnail_size: Tuple[int, int] = (200, 200)):
        """
        Initialize the MediaManager with specified configurations
        
        Args:
            output_dir (str): Directory for processed media files
            thumbnail_size (Tuple[int, int]): Default size for thumbnails
        """
        self.output_dir = output_dir
        self.thumbnail_size = thumbnail_size
        self.logger = logging.getLogger(__name__)
        
        # Create necessary directories
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "thumbnails"), exist_ok=True)
        
    def process_image(self, image_path: str) -> Dict:
        """
        Process an image file and generate thumbnails
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Dict containing processed image data and metadata
        """
        try:
            # Load and process image
            image = Image.open(image_path)
            
            # Generate thumbnail
            thumbnail = image.copy()
            thumbnail.thumbnail(self.thumbnail_size)
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            new_filename = f"{base_name}_{timestamp}"
            
            # Save processed files
            processed_path = os.path.join(self.output_dir, "images", f"{new_filename}.jpg")
            thumbnail_path = os.path.join(self.output_dir, "thumbnails", f"{new_filename}_thumb.jpg")
            
            image.save(processed_path, "JPEG")
            thumbnail.save(thumbnail_path, "JPEG")
            
            # Return metadata
            return {
                "original_path": image_path,
                "processed_path": processed_path,
                "thumbnail_path": thumbnail_path,
                "size": image.size,
                "format": image.format,
                "timestamp": timestamp
            }
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {str(e)}")
            return {"error": str(e)}
    
    def process_video(self, video_path: str, extract_frames: bool = False) -> Dict:
        """
        Process a video file and optionally extract frames
        
        Args:
            video_path (str): Path to the video file
            extract_frames (bool): Whether to extract frames from the video
            
        Returns:
            Dict containing processed video data and metadata
        """
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Generate thumbnail from first frame
            ret, frame = cap.read()
            if ret:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                new_filename = f"{base_name}_{timestamp}"
                
                # Save thumbnail
                thumbnail_path = os.path.join(self.output_dir, "thumbnails", f"{new_filename}_thumb.jpg")
                thumbnail = cv2.resize(frame, self.thumbnail_size)
                cv2.imwrite(thumbnail_path, thumbnail)
                
                # Extract frames if requested
                frames = []
                if extract_frames:
                    frames_dir = os.path.join(self.output_dir, "videos", new_filename, "frames")
                    os.makedirs(frames_dir, exist_ok=True)
                    
                    frame_interval = int(fps)  # Extract one frame per second
                    frame_count = 0
                    
                    while ret:
                        if frame_count % frame_interval == 0:
                            frame_path = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
                            cv2.imwrite(frame_path, frame)
                            frames.append(frame_path)
                        
                        ret, frame = cap.read()
                        frame_count += 1
            
            cap.release()
            
            # Return metadata
            return {
                "original_path": video_path,
                "thumbnail_path": thumbnail_path,
                "fps": fps,
                "frame_count": frame_count,
                "resolution": (width, height),
                "extracted_frames": frames if extract_frames else None,
                "timestamp": timestamp
            }
            
        except Exception as e:
            self.logger.error(f"Error processing video {video_path}: {str(e)}")
            return {"error": str(e)}
    
    def create_gallery(self, image_paths: List[str], output_path: str, 
                      cols: int = 3, padding: int = 10) -> str:
        """
        Create an image gallery from multiple images
        
        Args:
            image_paths (List[str]): List of paths to images
            output_path (str): Path to save the gallery
            cols (int): Number of columns in the gallery
            padding (int): Padding between images
            
        Returns:
            str: Path to the created gallery image
        """
        try:
            images = [Image.open(path) for path in image_paths]
            
            # Calculate gallery dimensions
            n_images = len(images)
            rows = (n_images + cols - 1) // cols
            
            # Calculate total size
            max_width = max(img.size[0] for img in images)
            max_height = max(img.size[1] for img in images)
            
            gallery_width = cols * (max_width + padding) - padding
            gallery_height = rows * (max_height + padding) - padding
            
            # Create gallery image
            gallery = Image.new('RGB', (gallery_width, gallery_height), 'white')
            
            # Place images
            for idx, img in enumerate(images):
                row = idx // cols
                col = idx % cols
                x = col * (max_width + padding)
                y = row * (max_height + padding)
                
                # Center image in its cell
                x_offset = (max_width - img.size[0]) // 2
                y_offset = (max_height - img.size[1]) // 2
                
                gallery.paste(img, (x + x_offset, y + y_offset))
            
            # Save gallery
            gallery.save(output_path, "JPEG")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error creating gallery: {str(e)}")
            return None
    
    def cleanup_old_files(self, max_age_days: int = 30):
        """
        Clean up old processed files
        
        Args:
            max_age_days (int): Maximum age of files to keep
        """
        try:
            current_time = datetime.now()
            
            for root, dirs, files in os.walk(self.output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                    
                    if (current_time - file_time).days > max_age_days:
                        os.remove(file_path)
                        self.logger.info(f"Removed old file: {file_path}")
                        
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")