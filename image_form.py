import cv2
import easyocr
import pandas as pd

# 1. Initialize EasyOCR Reader
# EasyOCR automatically detects and utilizes PyTorch/GPU if available.
# Set gpu=False if you do not have an Nvidia GPU configured with CUDA.
print("Initializing EasyOCR reader...")
reader = easyocr.Reader(['en'], gpu=True)

# Path to your image
image_path = 'unnamed.png'

# -------------------------------------------------------------
# APPROACH 1: Extract all text globally from the entire image
# -------------------------------------------------------------
print("\n--- Running Global Text Extraction ---")
results = reader.readtext(image_path)

# Parse and display the results
parsed_data = []
for (bbox, text, prob) in results:
    # bbox contains the 4 corner coordinates: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    print(f"Detected Text: {text} | Confidence: {prob:.2f}")
    parsed_data.append({
        "Text": text,
        "Confidence": prob,
        "BoundingBox": bbox
    })

# Save to a DataFrame for easy handling
df_all = pd.DataFrame(parsed_data)

# -------------------------------------------------------------
# APPROACH 2: Targeted Extraction (Best for Handwritten Fields)
# -------------------------------------------------------------
# Because handwriting inside specific form boxes can sometimes mix
# with printed lines, cropping the exact input fields yields higher accuracy.

print("\n--- Running Targeted Field Extraction ---")
img = cv2.imread(image_path)

# Define your regions of interest [ymin, ymax, xmin, xmax]
# Note: You will need to calibrate these coordinates based on your image's pixel dimensions.
fields_to_crop = {
    "Number of Accredited Voters": [300, 340, 730, 870],
    "Total Valid Votes": [480, 520, 730, 870],
    # Add coordinates for political party rows as needed
}

for field_name, coords in fields_to_crop.items():
    ymin, ymax, xmin, xmax = coords
    cropped_zone = img[ymin:ymax, xmin:xmax]

    # Run OCR strictly on the cropped handwritten box
    field_result = reader.readtext(cropped_zone)

    # Combine text if multiple snippets are found in the box
    extracted_text = " ".join([text for (_, text, _) in field_result])
    print(f"{field_name}: {extracted_text if extracted_text else '[No text detected]'}")