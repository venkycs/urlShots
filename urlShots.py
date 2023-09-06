from optimum.pipelines import pipeline
import onnxruntime as ort
ort.set_default_logger_severity(3)
import requests
from bs4 import BeautifulSoup
import html2text
import re

# Download AI model using command listed below
# optimum-cli export onnx -m venkycs/securityShots  --optimize O2 ./urlShots-model

def download_and_clean(url):
    try:
        headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:99.0) Gecko/20100101 Firefox/99.0"}
        response = requests.get(url,headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract the main body content (remove scripts, styles, etc.)
        for script in soup(["script", "style","img","a"]):
            script.extract()
        
        body_text = soup.get_text()
        
        # Use html2text to convert HTML to clean text
        h = html2text.HTML2Text()
        h.ignore_links = True  # Ignore hyperlinks
        h.ignore_images = True  # Ignore images
        h.ignore_emphasis = True  # Ignore emphasis (italics, bold)
        h.ignore_tables = True  # Ignore tables
        clean_text = h.handle(body_text)
        
        # Remove special characters, extra whitespace, and tabs
        clean_text = re.sub(r'[^\w\s]', '', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text
    
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None

onnx_qa = pipeline("summarization", model="urlShots-model/", accelerator="ort")
pred = onnx_qa(download_and_clean("https://thehackernews.com/2023/09/9-alarming-vulnerabilities-uncovered-in.html"))
print(pred)