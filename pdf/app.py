import pdfminer
from Scripts.pdf2txt import extract_text

text = extract_text("shahabuddin.pdf")
print(text)