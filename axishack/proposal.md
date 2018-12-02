# Idea
Contributor: Pratik Bhavsar

#### Proposed Features
 - Works with all kinds of PDFs and images
 - No manual intervention is required.
 - Finally saves output in the required format

### Solution Design
- Detect if the document is PDF or image
- Identify the type of pdf
    - Normal text pdf - Parsable pdf
    - Sandwich pdf - Image pdf with parsable text
    - Image pdf - Cannot be parsed for text
- Then extract the tables using OCR and non-OCR methods

### Tech Stack
- Python
- Python libraries
    - pdftabextract
    - opencv_python
    - pdf2image
    - pandas
    - numpy
    - PyPDF2
    - jupyterlab
    - scikit-learn
    - tabula-py
    - camelot
- [tesseract](https://github.com/tesseract-ocr/tesseract)

### Algorithms/Models
- A ML model to detect if the page contains a table
- Some image processing and NLP algorithms as required
