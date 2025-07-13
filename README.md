# Credit-Card-Private-Information-Remover
# EasyOCR Text Detection Program

A Python program for detecting and extracting text from images using EasyOCR with advanced features like pattern-based text redaction and multiple language support.

## Features

- **Basic Text Detection**: Extract text from images with confidence scores
- **Visual Annotation**: Display images with bounding boxes around detected text
- **Pattern-Based Redaction**: Automatically black out sensitive patterns (credit card numbers, dates, etc.)
- **Multiple Language Support**: Detect text in multiple languages simultaneously
- **Paragraph Detection**: Group text into logical paragraphs
- **File Output**: Save results to text files
- **Batch Processing**: Process multiple images at once

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Step 1: Install Required Packages

```bash
pip install easyocr opencv-python pillow matplotlib numpy
```

### Step 2: Download the Program

Save the Python code to a file named `easyocr_detector.py`

## Usage

### Basic Usage

1. **Place your image file** in the same directory as the script
2. **Update the image path** in the code:
   ```python
   image_path = "your_image_name.jpg"  # Replace with your image file
   ```
3. **Run the program**:
   ```bash
   python easyocr_detector.py
   ```

### What the Program Does

When you run the program, it will:

1. **Basic Text Detection** - Prints all detected text with confidence scores
2. **Visual Detection** - Shows an image with:
   - Green boxes around regular text
   - Black boxes covering sensitive patterns (numbers, dates)
3. **Paragraph Detection** - Groups text into logical paragraphs
4. **Multiple Language Detection** - Detects text in English, Spanish, and French
5. **Save Results** - Creates `ocr_results.txt` with all detected text (still in progress)

### Sensitive Pattern Detection

The program automatically detects and blacks out:
- 4-digit numbers (e.g., "1234")
- 3-digit numbers (e.g., "123")
- Date formats (e.g., "12/25", "01/2024")
- Special patterns (e.g., "123 45/67")

### Customization Options

#### Process Multiple Images
```python
# Add this to your main function:
image_paths = ["image1.jpg", "image2.png", "image3.jpg"]
all_results = process_multiple_images(image_paths)
```

### Performance Tips

- **First run is slow**: EasyOCR downloads language models
- **Subsequent runs are faster**: Models are cached locally
- **Better results**: Use high-resolution, well-lit images
- **Speed optimization**: Limit languages to only what you need

## Advanced Features

### Custom Pattern Detection
Modify the regex patterns in `detect_text_with_visualization()` to detect different sensitive information:

```python
# Add custom patterns
pattern_email = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
pattern_phone = r'^\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$'
```

### Batch Processing
Process multiple images at once:

```python
image_list = ["card1.jpg", "card2.jpg", "card3.jpg"]
batch_results = process_multiple_images(image_list)
```

## Requirements

- Python 3.7+
- easyocr
- opencv-python
- pillow
- matplotlib
- numpy
