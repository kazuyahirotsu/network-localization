import fitz  # PyMuPDF
import os

def convert_pdfs_to_text(pdf_dir):
    """
    Converts all PDF files in a directory to text files.
    """
    if not os.path.exists(pdf_dir):
        print(f"Directory not found: {pdf_dir}")
        return

    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            text_path = os.path.join(pdf_dir, os.path.splitext(filename)[0] + ".txt")

            try:
                print(f"Processing {pdf_path}...")
                doc = fitz.open(pdf_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                
                with open(text_path, "w", encoding="utf-8") as text_file:
                    text_file.write(text)
                print(f"Successfully converted to {text_path}")
            except Exception as e:
                print(f"Could not process {pdf_path}: {e}")

if __name__ == "__main__":
    pdf_directory = "paper/related work"
    convert_pdfs_to_text(pdf_directory)
