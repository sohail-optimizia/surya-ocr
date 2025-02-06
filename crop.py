import os
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
from rembg import remove
from PIL import Image
import io
import json
from typing import List
import pandas as pd
import streamlit as st
import pypdfium2
from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection
from surya.model.detection.segformer import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from surya.ocr import run_ocr
from surya.languages import CODE_TO_LANGUAGE
from surya.input.langs import replace_lang_with_code
from surya.schema import OCRResult

# Load recognition and detection models
@st.cache_resource()
def load_rec_cached():
    return load_rec_model(), load_rec_processor()

@st.cache_resource()
def load_det_cached():
    return load_det_model(), load_det_processor()

rec_model, rec_processor = load_rec_cached()
det_model, det_processor = load_det_cached()

# OCR processing function
def ocr(img, langs: List[str]) -> OCRResult:
    replace_lang_with_code(langs)
    img_pred = run_ocr([img], [langs], det_model=det_model, det_processor=det_processor, rec_model=rec_model, rec_processor=rec_processor)[0]
    return img_pred

# Functions for handling PDFs
def open_pdf(pdf_file):
    stream = io.BytesIO(pdf_file.read())
    return pypdfium2.PdfDocument(stream)

@st.cache_data()
def get_page_image(pdf_file, page_num, dpi=96):
    doc = open_pdf(pdf_file)
    renderer = doc.render(
        pypdfium2.PdfBitmap.to_pil,
        page_indices=[page_num - 1],
        scale=dpi / 72,
    )
    png = list(renderer)[0]
    png_image = png.convert("RGB")
    return png_image

@st.cache_data()
def get_pdf_page_count(pdf_file):
    doc = open_pdf(pdf_file)
    return len(doc)

# Process PDF files
def process_pdf(pdf_path, languages):
    with open(pdf_path, "rb") as pdf_file:
        doc = open_pdf(pdf_file)
        pil_image = get_page_image(pdf_file, 1)  # Process only the first page for simplicity
        ocr_result = ocr(pil_image, languages)
        results = {
            "filename": os.path.basename(pdf_path),
            "original_image": pil_image,
            "rotated_image": None,
            "cropped_image": None,
            "text": None,
            "ocr": ocr_result,
        }
    return results

# Correct orientation and remove background
def correct_orientation_and_remove_background(image: Image.Image) -> (Image.Image, Image.Image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    results = pytesseract.image_to_osd(gray, output_type=Output.DICT)
    angle = results.get("rotate", 0)

    if angle != 0:
        (h, w) = image_cv.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        rotated = cv2.warpAffine(image_cv, M, (new_w, new_h))
    else:
        rotated = image_cv

    rotated_pil = Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    output_image = remove(rotated_pil).convert("RGBA")
    data = np.array(output_image)
    alpha_channel = data[:, :, 3]
    alpha_threshold = 10
    mask = alpha_channel > alpha_threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return rotated_pil, rotated_pil

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    cropped_image = output_image.crop((xmin, ymin, xmax + 1, ymax + 1))
    return rotated_pil, cropped_image

# Process image files
def process_image(image_path, languages):
    pil_image = Image.open(image_path).convert("RGB")
    rotated_image, cropped_image = correct_orientation_and_remove_background(pil_image)
    ocr_result = ocr(cropped_image, languages)
    results = {
        "filename": os.path.basename(image_path),
        "original_image": pil_image,
        "rotated_image": rotated_image,
        "cropped_image": cropped_image,
        "text": None,
        "ocr": ocr_result,
    }
    return results

# Convert OCR results to CSV
def ocr_results_to_csv(results):
    rows = []
    for result in results:
        if result['ocr']:
            ocr_data = result['ocr'].text_lines
            for line in ocr_data:
                bbox = line.bbox
                if isinstance(bbox, list) and all(isinstance(point, (list, tuple)) and len(point) == 2 for point in bbox):
                    rows.append({
                        'filename': result['filename'],
                        'text': line.text,
                        'bbox_x1': bbox[0][0],
                        'bbox_y1': bbox[0][1],
                        'bbox_x2': bbox[1][0],
                        'bbox_y2': bbox[1][1],
                        'bbox_x3': bbox[2][0],
                        'bbox_y3': bbox[2][1],
                        'bbox_x4': bbox[3][0],
                        'bbox_y4': bbox[3][1],
                    })
                else:
                    rows.append({
                        'filename': result['filename'],
                        'text': line.text,
                        'bbox_x1': None,
                        'bbox_y1': None,
                        'bbox_x2': None,
                        'bbox_y2': None,
                        'bbox_x3': None,
                        'bbox_y3': None,
                        'bbox_x4': None,
                        'bbox_y4': None,
                    })
        elif result['text']:
            rows.append({
                'filename': result['filename'],
                'text': result['text'],
                'bbox_x1': None,
                'bbox_y1': None,
                'bbox_x2': None,
                'bbox_y2': None,
                'bbox_x3': None,
                'bbox_y3': None,
                'bbox_x4': None,
                'bbox_y4': None,
            })
    
    return pd.DataFrame(rows)

# Convert OCR results to JSON
def ocr_results_to_json(results):
    data = []
    for result in results:
        if result['ocr']:
            ocr_data = result['ocr'].text_lines
            for line in ocr_data:
                bbox = line.bbox
                if isinstance(bbox, list) and all(isinstance(point, (list, tuple)) and len(point) == 2 for point in bbox):
                    data.append({
                        'filename': result['filename'],
                        'text': line.text,
                        'bbox': bbox
                    })
                else:
                    data.append({
                        'filename': result['filename'],
                        'text': line.text,
                        'bbox': None
                    })
        elif result['text']:
            data.append({
                'filename': result['filename'],
                'text': result['text'],
                'bbox': None
            })
    
    return json.dumps(data, indent=4)

# Streamlit UI setup
st.title("Surya OCR Application")

languages = st.sidebar.multiselect(
    "Select OCR languages",
    options=list(CODE_TO_LANGUAGE.keys()),
    default=["en"]
)

# Path to the invoices folder
invoices_folder = "invoices"

# Get all files from the invoices folder
files = [os.path.join(invoices_folder, f) for f in os.listdir(invoices_folder) if f.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png'))]

if files:
    results = []
    for file in files:
        if file.lower().endswith(('jpg', 'jpeg', 'png')):
            results.append(process_image(file, languages))
        elif file.lower().endswith('pdf'):
            results.append(process_pdf(file, languages))
    
    if results:
        st.write(f"Processed {len(results)} files.")
        csv_data = ocr_results_to_csv(results)
        json_data = ocr_results_to_json(results)
        
        for result in results:
            st.subheader(f"Processed: {result['filename']}")
            
            # Display the original and cropped images
            st.image(result["original_image"], caption="Original Image", use_column_width=True)
            if result["cropped_image"]:
                st.image(result["cropped_image"], caption="Cropped Image", use_column_width=True)
                
                # Add a button to perform further actions on the cropped image
                if st.button(f"Perform OCR on {result['filename']}"):
                    # Example action: Re-run OCR or any other action you want
                    st.write(f"OCR performed again on {result['filename']}.")

        st.download_button(
            label="Download CSV",
            data=csv_data.to_csv(index=False),
            file_name="ocr_results.csv",
            mime="text/csv"
        )

        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name="ocr_results.json",
            mime="application/json"
        )
else:
    st.warning("No files found in the invoices folder.")
