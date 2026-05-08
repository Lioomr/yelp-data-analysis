import sys

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    pdf_path = sys.argv[1]
    
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            print(page.extract_text())
        return
    except ImportError:
        pass

    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            print(page.extract_text())
        return
    except ImportError:
        pass

    try:
        import fitz
        doc = fitz.open(pdf_path)
        for page in doc:
            print(page.get_text())
        return
    except ImportError:
        pass

    print("No PDF reading library found. Please install pypdf, PyPDF2, or pymupdf.")

if __name__ == '__main__':
    main()
