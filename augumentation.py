import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import os
import cv2
from pathlib import Path

class LoadFaces:
    """
    A class for loading and augmenting face images with various transformations
    """
    def __init__(self, directory):
        """
        Initialize the face loader with a directory path
        
        Args:
            directory (str): Path to the root directory containing face images
        """
        self.directory = Path(directory)
        self.output_dir = self.directory.parent / 'augmented'
        self.output_dir.mkdir(exist_ok=True)
        
        # Define augmentation sequence
        self.seq = iaa.Sequential([
            # Horizontal flip with 50% probability
            iaa.Fliplr(0.5),
            
            # Gaussian blur sometimes
            iaa.Sometimes(
                0.5,
                iaa.GaussianBlur(sigma=(0, 1))
            ),
            
            # Contrast adjustment
            iaa.LinearContrast((0.75, 1.5)),
            
            # Color multiplication
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            
            # Geometric transformations
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-30, 30),
                shear=(-10, 10)
            )
        ], random_order=True)

    def load_images_from_directory(self, directory):
        """
        Load all images from a directory
        
        Args:
            directory (Path): Directory path containing images
            
        Returns:
            tuple: (list of images, list of image names)
        """
        images = []
        image_names = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for img_path in directory.iterdir():
            if img_path.suffix.lower() in valid_extensions:
                try:
                    # Read image in BGR format
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        # Convert to RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img)
                        image_names.append(img_path.name)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        return images, image_names

    def augment_images(self, images, n_augmentations=5):
        """
        Generate multiple augmented versions of each image
        
        Args:
            images (list): List of input images
            n_augmentations (int): Number of augmented versions to generate per image
            
        Returns:
            list: Augmented images
        """
        augmented_images = []
        
        for _ in range(n_augmentations):
            aug_batch = self.seq(images=images)
            augmented_images.extend(aug_batch)
            
        return augmented_images

    def save_augmented_images(self, images, base_names, subdir_name):
        """
        Save augmented images to output directory
        
        Args:
            images (list): List of augmented images
            base_names (list): Original image names
            subdir_name (str): Name of the subject/class subdirectory
        """
        # Create subdirectory for this subject
        subject_dir = self.output_dir / subdir_name
        subject_dir.mkdir(exist_ok=True)
        
        # Save each augmented image
        for idx, img in enumerate(images):
            # Determine base name and augmentation number
            base_name = base_names[idx // 5]  # Integer division to get original image index
            aug_num = idx % 5  # Get augmentation number
            
            # Create output filename
            output_name = f"{Path(base_name).stem}_aug{aug_num}{Path(base_name).suffix}"
            output_path = subject_dir / output_name
            
            # Convert back to BGR for saving
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), img_bgr)

    def process_all_directories(self):
        """
        Process all subdirectories in the main directory
        """
        # Process each subject's directory
        for subdir in self.directory.iterdir():
            if subdir.is_dir() and subdir != self.output_dir:
                print(f"Processing directory: {subdir.name}")
                
                # Load images
                images, image_names = self.load_images_from_directory(subdir)
                if not images:
                    print(f"No valid images found in {subdir.name}")
                    continue
                
                print(f"Found {len(images)} images")
                
                # Generate augmented versions
                augmented_images = self.augment_images(images)
                print(f"Generated {len(augmented_images)} augmented images")
                
                # Save augmented images
                self.save_augmented_images(augmented_images, image_names, subdir.name)
                print(f"Saved augmented images for {subdir.name}")

# Usage example
if __name__ == "__main__":
    # Initialize the face loader
    aug = LoadFaces("faces_data/raw")
    
    # Process all directories
    aug.process_all_directories()