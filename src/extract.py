import os
import pymupdf
from docx import Document
def extract_text(path):
    extension = os.path.splitext(path)[1].lower()
    results = []
    if os.path.getsize(path) == 0:
        print(f"WARNING: Empty file - {path}")
        return results
    try:
        if extension == ".pdf":
            pdf = pymupdf.open(path)
            if pdf.is_encrypted:
                print(f"WARNING: Password protected PDF - {path}")
                pdf.close()
                return results
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if text:
                    results.append({
                        "page": page_number,
                        "text": text
                    })
            pdf.close()
            if not results:
                print(f"WARNING: Scanned PDF skipped - {path}")
        elif extension == ".docx":
            document = Document(path)
            for section_number, paragraph in enumerate(
                document.paragraphs, start=1
            ):
                text = paragraph.text.strip()
                if text:
                    results.append({
                        "page": section_number,
                        "text": text
                    })
        elif extension == ".txt":
            try:
                with open(path, "r", encoding="utf-8") as file:
                    text = file.read()
            except UnicodeDecodeError:
                with open(path, "r", encoding="latin-1") as file:
                    text = file.read()
                print(f"WARNING: Encoding fallback used - {path}")
            if text.strip():
                results.append({
                    "page": 1,
                    "text": text
                })
        else:
            print(f"Unsupported file type - {path}")
    except Exception as error:
        print(f"ERROR: {path} - {error}")
    return results
def extraction_summary(folder):
    supported = (".pdf", ".docx", ".txt")
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if not os.path.isfile(path):
            continue
        if not filename.lower().endswith(supported):
            continue
        pages = extract_text(path)
        characters = sum(
            len(page["text"]) for page in pages
        )
        print(
            f"File: {filename} | "
            f"Type: {os.path.splitext(filename)[1]} | "
            f"Pages/Sections: {len(pages)} | "
            f"Characters: {characters}"
        )
if __name__ == "__main__":
    extraction_summary("data")
