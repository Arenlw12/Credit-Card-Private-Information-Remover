import easyocr
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import re

def detect_text_basic(image_path, languages=['en']):
    # Initialize reader
    reader = easyocr.Reader(languages)

    # Detect text
    results = reader.readtext(image_path)

    print("Detected Text:")
    print("-" * 50)
    for i, (bbox, text, confidence) in enumerate(results):
        print(f"Text {i+1}: {text}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Bounding Box: {bbox}")
        print("-" * 30)
    
    return results

def detect_text_with_visualization(image_path, languages=['en']):
    """
    Text detection with bounding box visualization and blacking out for specific number patterns
    
    Args:
        image_path (str): Path to the image file
        languages (list): List of language codes
    
    Returns:
        tuple: (results, annotated_image)
    """
    reader = easyocr.Reader(languages)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file '{image_path}' not found.")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = reader.readtext(image_path)
    
    # Copy image for annotation
    annotated_image = image_rgb.copy()
    
    # Regex Patterns
    pattern_4_digits = r'^\d{4}$'
    pattern_3_digits = r'^\d{3}$'
    pattern_any_slash_any = r'^\d+/\d+$'
    pattern_unique = r'^\d+\s\d+/\d+$'
    
    margin = 10
    
    for (bbox, text, confidence) in results:
        # Extract coordinates
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        
        # Check if text matches any of the patterns
        if (re.match(pattern_4_digits, text) or 
            re.match(pattern_3_digits, text) or 
            re.match(pattern_any_slash_any, text) or
            re.match(pattern_unique, text)):
            # Calculate expanded region for blackout
            x, y = top_left
            w = bottom_right[0] - top_left[0]
            h = bottom_right[1] - top_left[1]
            
            # Expand the bounding box
            x_expanded = max(0, x - margin)
            y_expanded = max(0, y - margin)
            x2_expanded = min(annotated_image.shape[1], bottom_right[0] + margin)
            y2_expanded = min(annotated_image.shape[0], bottom_right[1] + margin)
            
            # Ensure valid region
            if (x2_expanded - x_expanded) > 0 and (y2_expanded - y_expanded) > 0:
                # Draw black rectangle
                cv2.rectangle(annotated_image, 
                            (x_expanded, y_expanded), 
                            (x2_expanded, y2_expanded), 
                            (0, 0, 0),
                            thickness=-1)
        else:
            # Draw rectangle for non-number text
            cv2.rectangle(annotated_image, top_left, bottom_right, (0, 255, 0), 2)
            # Add text label
            cv2.putText(annotated_image, f"{text} ({confidence:.2f})", 
                       (top_left[0], top_left[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return results, annotated_image

def detect_text_advanced(image_path, languages=['en'], 
                        paragraph=False, width_ths=0.7, height_ths=0.7):
    """
    Advanced text detection with custom parameters
    
    Args:
        image_path (str): Path to the image file
        languages (list): List of language codes
        paragraph (bool): Whether to group text into paragraphs
        width_ths (float): Width threshold for text grouping
        height_ths (float): Height threshold for text grouping
    
    Returns:
        list: Detected text results
    """
    # Initialize reader
    reader = easyocr.Reader(languages)
    
    # Read text with custom parameters
    results = reader.readtext(image_path, 
                             paragraph=paragraph,
                             width_ths=width_ths,
                             height_ths=height_ths)
    
    return results

def process_multiple_images(image_paths, languages=['en']):
    """
    Process multiple images for text detection
    
    Args:
        image_paths (list): List of image file paths
        languages (list): List of language codes
    
    Returns:
        dict: Dictionary with image paths as keys and results as values
    """
    reader = easyocr.Reader(languages)
    all_results = {}
    
    for image_path in image_paths:
        print(f"Processing: {image_path}")
        results = reader.readtext(image_path)
        all_results[image_path] = results
        
        # Print extracted text
        extracted_text = " ".join([text for _, text, _ in results])
        print(f"Extracted text: {extracted_text}")
        print("-" * 50)
    
    return all_results

def save_results_to_file(results, output_file):
    """
    Save OCR results to a text file
    
    Args:
        results (list): OCR results from EasyOCR
        output_file (str): Path to output text file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (bbox, text, confidence) in enumerate(results):
            f.write(f"Text {i+1}: {text}\n")
            f.write(f"Confidence: {confidence:.2f}\n")
            f.write(f"Bounding Box: {bbox}\n")
            f.write("-" * 30 + "\n")

def main():
    """
    Main function demonstrating different EasyOCR usage examples
    """
    # Example image path - replace with your image
    image_path = "vip-membership-card3.jpg"
    
    try:
        # Basic text detection
        print("=== Basic Text Detection ===")
        results = detect_text_basic(image_path)
        
        # Text detection with visualization
        print("\n=== Text Detection with Visualization ===")
        results, annotated_img = detect_text_with_visualization(image_path)
        
        # Display the annotated image
        plt.figure(figsize=(12, 8))
        plt.imshow(annotated_img)
        plt.axis('off')
        plt.title('Text Detection Results')
        plt.show()
        
        # Advanced detection with paragraphs
        print("\n=== Advanced Detection (Paragraphs) ===")
        paragraph_results = detect_text_advanced(image_path, paragraph=True)
        
        for i, (bbox, text, confidence) in enumerate(paragraph_results):
            print(f"Paragraph {i+1}: {text}")
            print(f"Confidence: {confidence:.2f}")
            print("-" * 30)
        
        # Save results to file
        save_results_to_file(results, "ocr_results.txt")
        print("\nResults saved to ocr_results.txt")
        
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        print("Please update the image_path variable with a valid image file.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()