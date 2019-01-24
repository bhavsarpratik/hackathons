# PDF Do HTML Lo
This was the winning solution 

Team: SoloHeToKyaHua
Contributor: Pratik Bhavsar

### Code Flow
- Identify the type of pdf
    - Normal text pdf - Parsable pdf
    - Sandwich pdf - Image pdf with parsable text
    - Image pdf - Cannot be parsed for text
- Then it finds the pages containing the required tables using Multinomial Naive Bayes
    - Balance sheet
    - Income statement
    - Cash-flow statement
- Then it does a combination of pdf-image, image-ocrpdf and ocrpdf-html to extract the tables

### /scripts
Contains scripts to be called
  - pdf_to_html.py - Contains the logic to generate HTML from PDF
  - config.py - Contains settings used by pdf_to_html.py
### /test
  - Put all PDFs to be tested in this folder
### /outputs
  - This will contain directories with the name of PDFs which are created on running pdf_to_html.py
### /outputs/pdfname/tables-and-html
 - This will contain the final table outputs in csv and html
### /outputs/pdfname/html-extra
 - This will contain extra html outputs
### /outputs/pdfname/table_pages_pdftype_xxx.csv
 - This file contains the pages which were detected for the required tables - Balance sheet, Income statement and Cash-flow statement along with mention of 'consolidated' or 'is note'. This are the probable pages containing our tables and hence they are extracted later.

| page |  prob |     Table type    | consolidated | is_note |
|:----:|:-----:|:-----------------:|:------------:|:-------:|
|  94  | 0.997 |  IncomeStatement  |     FALSE    |  FALSE  |
|  134 | 0.995 |  IncomeStatement  |     TRUE     |  FALSE  |
|  136 | 0.976 | CashflowStatement |     TRUE     |  FALSE  |
|  95  | 0.968 | CashflowStatement |     FALSE    |  FALSE  |
|  96  | 0.955 | CashflowStatement |     FALSE    |  FALSE  |
|  135 | 0.908 | CashflowStatement |     TRUE     |  FALSE  |
|  93  | 0.987 |  BalanceStatement |     FALSE    |  FALSE  |
|  133 | 0.986 |  BalanceStatement |     TRUE     |  FALSE  |
### /models
 - Contains trained NLP models used by pdf_to_html.py
### /notebooks
 - Contains notebooks used for testing and training NLP model
 ### /logs
 - Contains logs created while processing the PDF
# Setup
#### Environment setup on Windows

## Softwares
 Open this [link](https://morningstaronline-my.sharepoint.com/:f:/g/personal/pratik_bhavsar_morningstar_com/Eog-oAjbb-dMkCkzlM7I5RkBm2fX2x6V7X-0trlzWMzwVg?e=gza7Bu) and follow the below instructions

- ### Tesseract
Run tesseract-ocr exe to install it
Put this in environment path
`C:\Program Files (x86)\Tesseract-OCR`

- ### Poppler
Download and extract poppler zip to any path after downloading.
Put the path to extracted poppler's **bin** folder in environment path
C:\xxxxxxxxxx\poppler-0.67.0\bin\

- ### Python packges
Download and install python from the opened link
Then cd into the repo and run this on cmd
```
py -3.5 -m pip install virtualenv
py -3.5 -m virtualenv mshack
mshack\Scripts\activate #This activates the environment
pip install -r requirements.txt
```
`cd scripts` # This contains the executables
`python pdf_to_html.py #This runs the main script`
