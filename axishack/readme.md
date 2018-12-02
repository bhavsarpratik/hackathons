# PDF table extractor
Contributor: Pratik Bhavsar  
Environment: Windows 64Bit (tested for Windows 10)  
Written in Python

## Installation

- This setup is for Windows 64Bit
- Download and install all softwares from this [link](https://drive.google.com/open?id=1E7EQbbPFoJ7OCTmELQtKywdOlVxMRA58)
- While installing python select tick on **put path in environment**

- Put installation paths of tesseract, poppler and ghostscript in environment path   

```
C:\Program Files (x86)\Tesseract-OCR     
C:\Program Files (x86)\Tesseract-OCR\tessdata  
C:\Program Files\gs\gs9.26\bin   
Extract Poppler anywhere and put it's **bin** path i.e C:\xxxxxxxxxx\poppler-0.67.0\bin\  
```
Java - Tesseract requires the specific java to bin in path. Please check for errors when running.  
Put the required java in path.

- ### Python packges
Download and install python from the opened link.    
Then cd into the repo and run this on cmd
```
py -3.5 -m pip install virtualenv
py -3.5 -m virtualenv pratik-env
pratik-env\Scripts\activate #This activates the environment
pip install -r requirements.txt
```
## Running script
`Put the image/PDF to be parsed in test folder and the outputs can be found in outputs folder in a folder with the name of image/PDF`
```
cd repo
pratik-env\Scripts\activate #This activates the environment
cd scripts # This contains the executables
python main.py #This runs the main script
```
