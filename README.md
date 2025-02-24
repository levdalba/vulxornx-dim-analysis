# Vulxornx Dimension Analysis

This project provides a solution for analyzing technical engineering drawings (`.tif` files) to extract minimum and maximum dimensions in millimeters, targeting small/medium mechanical parts like rails. It is designed for the AI Engineer test at Vulcorn X, focusing on dimension detection using classical computer vision and OCR.

## Project Goal

The goal is to identify the smallest and largest dimensions (in millimeters) from engineering drawings, producing an output file (`predictions.csv`) with `drawing_file`, `min_dimension`, and `max_dimension` for each drawing. The solution runs on unseen data, ensuring scalability and accuracy for new `.tif` files.

## Installation

### Prerequisites

-   Python 3.9+ (recommended: 3.11)
-   Tesseract OCR (for text recognition):
    -   **macOS**: `brew install tesseract`
    -   **Ubuntu/Linux**: `sudo apt-get install tesseract-ocr`
    -   **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki) and install, updating `pytesseract.tesseract_cmd` in `main.py` if needed.

### Setup

1. Clone this repository:

    ```bash
    git clone <your-repo-url>
    cd vulxornx-dim-analysis
    ```

2. Create a virtual environment:

    ```bash
    python3 -m venv myenv
    ```

3. Activate the virtual environment:

    - **macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```
    - **Windows**:
        ```bash
        venv\Scripts\activate
        ```

4. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

5. Update the path to the Tesseract executable in `main.py` if needed:

    ```python
    pytesseract.pytesseract.tesseract_cmd = r"/opt/homebrew/bin/tesseract"
    ```

6. Update the input folder path in `main.py` if needed:

    ```python
    if __name__ == "__main__":
        input_folder = "drive-download-data"  # Update this path as needed
        main(input_folder, scale_factor=0.1)
    ```

7. Deactivate the virtual environment when done:
    ```bash
    deactivate
    ```
