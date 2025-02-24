import os
import cv2
import pytesseract
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, List, Optional
from tqdm import tqdm

MIN_VALID_DIM = 0.01
MAX_VALID_DIM = 50.0

# Update path to Tesseract executable if needed
pytesseract.pytesseract.tesseract_cmd = r"/opt/homebrew/bin/tesseract"


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Preprocess the image for better dimension detection.
    Converts to grayscale, applies adaptive thresholding,
    and cleans noise with morphological operations.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return processed


def detect_dimension_lines(image: np.ndarray, scale_factor: float = 0.1) -> List[float]:
    """
    Detect dimension lines using Canny edge detection and the Hough Transform.
    Only lines with angles near 0°, 90°, or 180° and lengths within the valid range are kept.
    """
    edges = cv2.Canny(image, 100, 200)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=30, maxLineGap=10
    )
    valid_line_lengths = []
    if lines is None:
        return valid_line_lengths

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate the line length in mm
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * scale_factor
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Accept lines that are approximately horizontal or vertical
        if abs(angle) < 10 or abs(abs(angle) - 90) < 10 or abs(abs(angle) - 180) < 10:
            if MIN_VALID_DIM <= length <= MAX_VALID_DIM:
                valid_line_lengths.append(length)
    return valid_line_lengths


def extract_text_dimensions(image: np.ndarray) -> List[float]:
    text = pytesseract.image_to_string(image, config="--psm 6 --oem 3", lang="eng+deu")
    print(f"Extracted text: {text}")

    numbers = []
    for word in text.split():
        try:
            num = float(word.replace(",", "."))  # Handle decimal notation
            if (
                0.05 <= num <= 50.0
                and ("." in word or "," in word)
                and len(word.split(".")[0]) <= 2
            ):
                if num < 2.0 or len(numbers) < 3:  # Limit larger min values
                    numbers.append(num)
                    print(f"Detected text dimension (mm): {num:.2f}")
        except ValueError:
            continue
    return numbers


def process_drawing(
    image_path: str, scale_factor: float = 1.0
) -> Tuple[Optional[float], Optional[float]]:
    try:
        processed_image = preprocess_image(image_path)
        line_dims = detect_dimension_lines(processed_image, scale_factor)
        text_dims = extract_text_dimensions(processed_image)

        print(f"Line dimensions for {image_path}: {line_dims}")
        print(f"Text dimensions for {image_path}: {text_dims}")

        all_dims = [d for d in line_dims + text_dims if 0.05 <= d <= 200.0]
        if not all_dims:
            print(f"No valid dimensions detected in {image_path}")
            return None, None

        min_dim = min(all_dims)
        max_dim = max(all_dims)
        print(
            f"Final dimensions for {image_path}: Min = {min_dim:.2f} mm, Max = {max_dim:.2f} mm"
        )
        return min_dim, max_dim
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None, None


def main(
    input_folder: str, scale_factor: float = 1.0, output_file: str = "predictions.csv"
) -> None:
    drawing_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".tif")]
    if not drawing_files:
        print(f"No .tif files found in {input_folder}")
        return

    results = []
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_drawing, os.path.join(input_folder, file), scale_factor
            ): file
            for file in drawing_files
        }
        for future in tqdm(futures, total=len(futures), desc="Processing drawings"):
            drawing_file = futures[future]
            print(f"Processing: {drawing_file}")
            min_dim, max_dim = future.result()
            results.append(
                {
                    "drawing_file": drawing_file,
                    "min_dimension": min_dim,
                    "max_dimension": max_dim,
                }
            )

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    input_folder = "drive-download-data"  # Update this path as needed
    main(input_folder, scale_factor=0.1)
