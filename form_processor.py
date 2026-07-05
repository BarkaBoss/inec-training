import os
import csv
import numpy as np
import pandas as pd
from PIL import Image
from pdf2image import convert_from_path, convert_from_bytes
import easyocr


def parse_ocr_spatially(ocr_results):
    """
    Parses handwritten form data by checking bounding boxes geometry.
    Looks vertically UPWARD for stacked fields (Name, Address, Occupation)
    and handles inline text fields.
    """
    extracted = {
        "Candidate Name": "Not Found",
        "Address": "Not Found",
        "Occupation": "Not Found",
        "Voter Identification Number": "Not Found",
        "Local Government/Area Council": "Not Found",
        "Ward": "Not Found",
        "Delimitation": "Not Found",
        "Sponsoring Party": "Not Found"
    }

    # Extract clean boundaries for every bounding box detected
    parsed_boxes = []
    for box, text, conf in ocr_results:
        x_min = min(pt[0] for pt in box)
        x_max = max(pt[0] for pt in box)
        y_min = min(pt[1] for pt in box)
        y_max = max(pt[1] for pt in box)
        center_x = (x_min + x_max) / 2
        parsed_boxes.append({
            "text": text.strip(),
            "y_min": y_min,
            "y_max": y_max,
            "center_x": center_x,
            "x_min": x_min,
            "x_max": x_max
        })

    for i, item in enumerate(parsed_boxes):
        text_lower = item["text"].lower()

        # --- RULE 1: LOOK UP FOR ADDRESS ---
        if "(address)" in text_lower:
            # Address text sits directly above the "(Address)" label template box
            candidates = []
            for candidate in parsed_boxes:
                # Must be higher on the page (y is smaller) and horizontally aligned
                if candidate["y_max"] <= item["y_min"] and abs(candidate["center_x"] - item["center_x"]) < 200:
                    # Ignore the global title "I," or "of" template markings
                    if candidate["text"].lower() not in ["1,", "i,", "of"]:
                        candidates.append(candidate)

            # Sort by proximity (closest vertical box above the label)
            if candidates:
                candidates.sort(key=lambda c: item["y_min"] - c["y_max"])
                extracted["Address"] = candidates[0]["text"]

                # Dynamic Check: The Candidate Name often sits directly above the Address box
                if len(candidates) > 1:
                    extracted["Candidate Name"] = candidates[1]["text"]

        # --- RULE 2: LOOK UP FOR OCCUPATION ---
        elif "(occupation)" in text_lower:
            candidates = []
            for candidate in parsed_boxes:
                if candidate["y_max"] <= item["y_min"] and abs(candidate["center_x"] - item["center_x"]) < 200:
                    if "hereby state" not in candidate["text"].lower():
                        candidates.append(candidate)
            if candidates:
                candidates.sort(key=lambda c: item["y_min"] - c["y_max"])
                extracted["Occupation"] = candidates[0]["text"]

        # --- RULE 3: INLINE INFERENCE (Text containing specific handwritten patterns) ---
        elif "voter identification" in text_lower or "vin" in text_lower:
            # Extracts matching digit sequences trailing nearby
            for candidate in parsed_boxes:
                if abs(candidate["y_min"] - item["y_min"]) < 30 and candidate["text"] != item["text"]:
                    extracted["Voter Identification Number"] = candidate["text"]

        elif "local government" in text_lower:
            # Look for adjacent string matches like 'AMAC AREA COUNCIL'
            for next_item in parsed_boxes:
                if abs(next_item["y_min"] - item["y_min"]) < 40 and next_item["center_x"] > item["center_x"]:
                    extracted["Local Government/Area Council"] = next_item["text"]

        elif "ward" in text_lower:
            # Looks for values adjacent to or immediately following 'Ward'
            if i + 1 < len(parsed_boxes):
                extracted["Ward"] = parsed_boxes[i + 1]["text"].replace(".", "").strip()

        elif "delimitation" in text_lower:
            if i + 1 < len(parsed_boxes):
                extracted["Delimitation"] = parsed_boxes[i + 1]["text"]

        elif "sponsored by" in text_lower:
            if i + 1 < len(parsed_boxes):
                extracted["Sponsoring Party"] = parsed_boxes[i + 1]["text"]

    return extracted


def process_scanned_form(pdf_path, csv_path="form_extractions.csv"):
    # Convert pdf structure
    images = convert_from_path(pdf_path)
    if not images:
        return

    # Process the target template sheet
    img_np = np.array(images[0])

    # Initialize the engine
    reader = easyocr.Reader(['en'], gpu=True)  # Swap to gpu=False if running without Apple Silicon/CUDA
    ocr_results = reader.readtext(img_np)

    # Execute structural coordinate evaluation
    form_data = parse_ocr_spatially(ocr_results)

    # Export immediately to Dataframe/CSV
    df = pd.DataFrame([form_data])
    file_exists = os.path.isfile(csv_path)
    df.to_csv(csv_path, mode='a', index=False, header=not file_exists, encoding='utf-8')

    return form_data


# --- Execution ---
if __name__ == "__main__":
    pdf_file_path = os.path.join("form", "Form-EC13A-1.pdf")
    if os.path.exists(pdf_file_path):
        results = process_scanned_form(pdf_file_path)
        print("\n--- Spatially Extracted Handwritten Values ---")
        for key, val in results.items():
            print(f"{key}: {val}")
    else:
        print(f"File target missing at: {pdf_file_path}")