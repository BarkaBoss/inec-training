import os
from PIL import Image
from pdf2image import convert_from_path
import easyocr


def convert_pdf_to_images(pdf_path):
    """Converts a PDF file into a list of PIL Images."""
    print(f"Converting {pdf_path} to images...")
    return convert_from_path(pdf_path)


def perform_ocr_on_image(image_path_or_buf):
    """
    Uses EasyOCR (PyTorch-based) to extract text and spatial bounding boxes.
    """
    # Initialize the reader with English. It automatically loads PyTorch weights.
    reader = easyocr.Reader(['en'], gpu=True)  # Set gpu=False if you don't have a CUDA GPU

    print("Running PyTorch-based EasyOCR extraction...")
    # easyocr accepts image paths, bytes, or PIL images directly
    results = reader.readtext(image_path_or_buf)

    return results


def parse_form_fields(ocr_results):
    """
    Heuristic parser that looks for form keys and attempts to find
    associated values based on textual layout or sequence.
    """
    extracted_data = {
        "Candidate Name": None,
        "Address": None,
        "Occupation": None,
        "Voter Identification Number": None,
        "Local Government/Area Council": None,
        "Ward": None,
        "Delimitation": None,
        "Educational Qualifications": None,
        "Vice President Nominee": None,
        "Sponsoring Party": None
    }

    text_lines = [res[1].strip() for res in ocr_results]

    for i, line in enumerate(text_lines):
        # Example Matching for Voter ID
        if "Voter Identification" in line or "VIN" in line:
            # Often the filled value is the next detected block or on the same line
            if i + 1 < len(text_lines):
                extracted_data["Voter Identification Number"] = text_lines[i + 1]

        elif "Local Government" in line:
            if i + 1 < len(text_lines):
                extracted_data["Local Government/Area Council"] = text_lines[i + 1]

        elif "Ward" in line:
            if i + 1 < len(text_lines):
                extracted_data["Ward"] = text_lines[i + 1]

        elif "Delimitation" in line:
            if i + 1 < len(text_lines):
                extracted_data["Delimitation"] = text_lines[i + 1]

        elif "sponsored by" in line.lower():
            if i + 1 < len(text_lines):
                extracted_data["Sponsoring Party"] = text_lines[i + 1]

    return extracted_data


def process_form(pdf_path):
    images = convert_pdf_to_images(pdf_path)

    if not images:
        print("No pages found in PDF.")
        return

    # Processing the first page (Form EC 13A is typically a multi-page document but it has been stripped to a single page)
    first_page = images[0]

    # Save temporarily to pass to OCR, or pass the PIL object directly
    temp_img_path = "temp_page1.png"
    first_page.save(temp_img_path, "PNG")

    ocr_results = perform_ocr_on_image(temp_img_path)

    form_data = parse_form_fields(ocr_results)

    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)

    return form_data


if __name__ == "__main__":
    pdf_file_path = "form/Form-EC13A.pdf"

    try:
        data = process_form(pdf_file_path)
        print("\n--- Extracted Form Data ---")
        for key, value in data.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"An error occurred: {e}")