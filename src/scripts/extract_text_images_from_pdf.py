
import sys
from pathlib import Path
import pytesseract
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
from fpdf import FPDF

pytesseract.pytesseract.tesseract_cmd = "C:/Users/1053914/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"

def check_tesseract_installation() -> bool:
    """Check if Tesseract OCR is properly installed.

    Returns:
        bool: True if Tesseract is available, False otherwise
    """
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        print("""
Error: Tesseract OCR is not installed or not in PATH. Please:
1. Download Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install it (make note of installation directory)
3. Add the installation directory to your system PATH
   OR
   Add this line at the start of the script:
   pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\1053914\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'
        """)
        return False

def check_poppler_installation() -> bool:
    """Check if Poppler is properly installed.

    Returns:
        bool: True if a test PDF conversion works, False otherwise
    """
    try:
        # Specify the path to your extracted Poppler bin directory
        poppler_path: str = "./poppler/Library/bin"  # Update this path to where you extracted Poppler
        
        # Try to convert first page of the PDF
        convert_from_path("./docs/doc-reduit.pdf", first_page=1, last_page=1, poppler_path=poppler_path)
        return True
    except PDFInfoNotInstalledError:
        print("""
Error: Poppler is not found. Please:
1. Download Poppler from: https://github.com/oschwartz10612/poppler-windows/releases/
2. Extract it to a folder (e.g., './poppler')
3. Update the poppler_path variable in the code to point to the 'bin' directory
        """)
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def extract_text_from_pdf(pdf_path: str, output_pdf_path: str, output_txt_path: str) -> None:
    """Extract text from PDF using OCR and save as both PDF and TXT.

    Args:
        pdf_path (str): Path to the input PDF file
        output_pdf_path (str): Path where to save the extracted text as PDF
        output_txt_path (str): Path where to save the extracted text as TXT
    """
    try:
        # Specify the path to your extracted Poppler bin directory
        poppler_path: str = "./poppler/Library/bin"  # Update this path to where you extracted Poppler
        
        # Convert PDF to images
        images = convert_from_path(pdf_path, poppler_path=poppler_path)
        
        # Perform OCR on each page
        extracted_text: str = "\n\n".join(
            pytesseract.image_to_string(image, lang="eng+fra") 
            for image in images
        )

        # Save as TXT file
        with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(extracted_text)

        # Create PDF
        pdf: FPDF = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Split text into lines and add to PDF, handling encoding
        for line in extracted_text.split('\n'):
            # Replace problematic characters and encode as ASCII
            clean_line = line.encode('ascii', 'replace').decode('ascii')
            pdf.cell(0, 10, clean_line, ln=True)
            
        # Save as PDF with UTF-8 encoding
        pdf.output(output_pdf_path, 'F')
            
        print(f"Text successfully extracted and saved as PDF to: {output_pdf_path}")
        print(f"Text also saved as TXT to: {output_txt_path}")
            
    except Exception as e:
        print(f"Error during text extraction: {e}")
        sys.exit(1)

def main(
    pdf_path: str = "./docs/doc-reduit.pdf",
    output_pdf_path: str = "./docs/extracted_text.pdf", 
    output_txt_path: str = "./docs/extracted_text.txt"
) -> None:
    """Main function to handle PDF text extraction.
    
    Args:
        pdf_path (str): Path to the input PDF file. Defaults to "./docs/doc-reduit.pdf"
        output_pdf_path (str): Path where to save the extracted text as PDF. Defaults to "./docs/extracted_text.pdf"
        output_txt_path (str): Path where to save the extracted text as TXT. Defaults to "./docs/extracted_text.txt"
    """
    # Ensure input PDF exists
    if not Path(pdf_path).exists():
        print(f"Error: PDF file not found at {pdf_path}")
        sys.exit(1)
    
    # Check dependencies
    if not check_tesseract_installation():
        sys.exit(1)
    if not check_poppler_installation():
        sys.exit(1)
        
    # Extract text and save as PDF and TXT
    extract_text_from_pdf(pdf_path, output_pdf_path, output_txt_path)

if __name__ == "__main__":
    main()
