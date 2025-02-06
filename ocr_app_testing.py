import os
import argparse
import io
from typing import List
import json
from xmlrpc import client as xmlrpclib
import pytesseract
import cv2
from helper.template2 import extract_invoice_data_2
import pypdfium2
import streamlit as st
from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection
from surya.model.detection.segformer import load_model, load_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from surya.model.ordering.processor import load_processor as load_order_processor
from surya.model.ordering.model import load_model as load_order_model
from surya.ordering import batch_ordering
from surya.postprocessing.heatmap import draw_polys_on_image
from surya.ocr import run_ocr
from surya.postprocessing.text import draw_text_on_image
from PIL import Image
from surya.languages import CODE_TO_LANGUAGE
from surya.input.langs import replace_lang_with_code
from surya.schema import OCRResult, TextDetectionResult, LayoutResult, OrderResult
from surya.settings import settings
import re
import numpy as np
from datetime import datetime
from xmlrpc import client as xmlrpclib

parser = argparse.ArgumentParser(description="Run OCR on an image or PDF.")
parser.add_argument("--math", action="store_true", help="Use math model for detection", default=False)


# Odoo configurations
ODOO_URL = 'https://odoo.itrivers.com' #/https://devo-optimizia.odoo.com
ODOO_DB = 'restaurant' #devo-optimizia
ODOO_USERNAME = 'demo@itrivers.sa' #developer@optimizia.co
ODOO_PASSWORD = 'demo@@' #cANCEL123@@
ODOO_MODEL = 'account.move'  # Replace 'your_model_name' with the actual model name

# Connect to Odoo
common = xmlrpclib.ServerProxy('{}/xmlrpc/2/common'.format(ODOO_URL))
uid = common.authenticate(ODOO_DB, ODOO_USERNAME, ODOO_PASSWORD, {})

# Search for existing record in Odoo
def search_existing_record(search_domain):
    models = xmlrpclib.ServerProxy('{}/xmlrpc/2/object'.format(ODOO_URL))
    existing_record_ids = models.execute_kw(ODOO_DB, uid, ODOO_PASSWORD, ODOO_MODEL, 'search', [search_domain])
    return existing_record_ids

# Check if a record exists in Odoo
def record_exists_odoo(data):
    search_domain = [['name', '=', data['name']]]
    existing_record_ids = search_existing_record(search_domain)
    return len(existing_record_ids) > 0

from decimal import Decimal

def clean_numeric_string(value):
    # Remove any characters that are not digits or a decimal point
    return ''.join(char for char in value if char.isdigit() or char == '.')

def process_invoices(json_data):
    models = xmlrpclib.ServerProxy('{}/xmlrpc/2/object'.format(ODOO_URL))
    invoices_data = json_data["results"]

    for invoice_data in invoices_data:
        invoice_inline = []
        extracted_details = invoice_data['extracted details']
        name = extracted_details['Invoice No. (رقم الفاتورة)']
        buyer_name = extracted_details['Customer (العميل)']

        # Check if the invoice already exists
        if record_exists_odoo({'name': name}):
            print(f"Invoice {name} already exists. Skipping...")
            continue

        # Extract relevant information for each invoice
        invoice_date_due = extracted_details['Due Date (تاريخ الاستحقاق)']
        invoice_date = extracted_details.get('Invoice Date (تاريخ الفاتورة)', '')  # Handle case when Invoice Date is not found
        discount = extracted_details.get('Discount (مجموع الخصم)', '')
        other_details = extracted_details.get("Notes", '')

        # Convert dates to Odoo's expected format %Y-%m-%d
        try:
            invoice_date_due = datetime.strptime(invoice_date_due, '%Y/%m/%d').strftime('%Y-%m-%d')
        except ValueError:
            invoice_date_due = False  # Handle invalid date format gracefully
        
        if invoice_date:
            try:
                invoice_date = datetime.strptime(invoice_date, '%Y/%m/%d').strftime('%Y-%m-%d')
            except ValueError:
                invoice_date = False  # Handle invalid date format gracefully

        # Extract products
        for product_info in extracted_details.get('Products Info', []):
            product_name = product_info.get('Nature of Goods or service (تفاصيل السلع أو الخدمات)', '')
            product_quantity = product_info.get('Quantity (الكمية)', '')
            product_description = product_info.get('Description وصف', '')

            # Handle list price with commas and dots
            list_price_str = product_info.get('Price Unit (\u0633\u0639\u0631 \u0627\u0644\u0648\u062d\u062f\u0629)', '0.0')
            list_price_str = list_price_str.replace(',', '')  # Remove commas
            try:
                list_price = float(list_price_str)  # Convert to float
            except ValueError:
                list_price = 0.0  # Default to 0.0 if conversion fails

            product_data = {
                "name": product_name +' ('+product_description+")",
                "list_price": list_price
            }
            
            # Search for existing product in Odoo
            product_id = models.execute_kw(ODOO_DB, uid, ODOO_PASSWORD, 'product.template', 'search', [[('name', '=', product_name)]], {'limit': 1})

            if not product_id:
                # Create product if not exists
                product_id = models.execute_kw(ODOO_DB, uid, ODOO_PASSWORD, 'product.template', 'create', [product_data])
            else:
                print("Product already exists.")

            # Append product line to invoice_inline
            invoice_inline.append([0, 0, {
                "product_id": product_id[0] if isinstance(product_id, list) else product_id,
                "quantity": str(product_quantity),
                "discount": str(discount)
            }])
        
        buyer_data = {
            'name': buyer_name, 
            "is_company": True,
            "street": "",
            "zip": "",
            "city": "",
            "country_id": 1,
            "vat": "" ,
            "phone": ""
         }


                # # Check if the buyer exists in Odoo
        buyer_id = models.execute_kw(ODOO_DB, uid, ODOO_PASSWORD, 'res.partner', 'search', [[('name', '=', buyer_name)]], {'limit': 1})
        if not buyer_id:
            # Create buyer if not exists
            buyer_id = models.execute_kw(ODOO_DB, uid, ODOO_PASSWORD, 'res.partner', 'create', [buyer_data])

        # Prepare invoice data to create in Odoo
        invoice_data = {
            "name": str(name),
            "display_name": str(name),
            "invoice_user_id": False,
            "partner_id": buyer_id if isinstance(buyer_id, int) else buyer_id[0],
            "move_type": "out_invoice",
            "payment_reference": "IMMEDIATE",
            "invoice_date_due": invoice_date_due if invoice_date_due else False,
            "invoice_date": invoice_date if invoice_date else False,
            "invoice_line_ids": invoice_inline
        }

        # Create the invoice record in Odoo
        try:
            invoice_id = models.execute_kw(ODOO_DB, uid, ODOO_PASSWORD, 'account.move', 'create', [invoice_data])
            print("Invoice created with ID:", invoice_id)
        except Exception as e:
            print(f"Failed to create invoice {name}: {str(e)}")

try:
    args = parser.parse_args()
except SystemExit as e:
    print(f"Error parsing arguments: {e}")
    os._exit(e.code)

@st.cache_resource()
def load_det_cached():
    checkpoint = settings.DETECTOR_MATH_MODEL_CHECKPOINT if args.math else settings.DETECTOR_MODEL_CHECKPOINT
    return load_model(checkpoint=checkpoint), load_processor(checkpoint=checkpoint)

@st.cache_resource()
def load_rec_cached():
    return load_rec_model(), load_rec_processor()

@st.cache_resource()
def load_layout_cached():
    return load_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT), load_processor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)

@st.cache_resource()
def load_order_cached():
    return load_order_model(), load_order_processor()

def text_detection(img) -> (Image.Image, TextDetectionResult):
    pred = batch_text_detection([img], det_model, det_processor)[0]
    polygons = [p.polygon for p in pred.bboxes]
    det_img = draw_polys_on_image(polygons, img.copy())
    return det_img, pred

def layout_detection(img) -> (Image.Image, LayoutResult):
    _, det_pred = text_detection(img)
    pred = batch_layout_detection([img], layout_model, layout_processor, [det_pred])[0]
    polygons = [p.polygon for p in pred.bboxes]
    labels = [p.label for p in pred.bboxes]
    layout_img = draw_polys_on_image(polygons, img.copy(), labels=labels)
    return layout_img, pred

def order_detection(img) -> (Image.Image, OrderResult):
    _, layout_pred = layout_detection(img)
    bboxes = [l.bbox for l in layout_pred.bboxes]
    pred = batch_ordering([img], [bboxes], order_model, order_processor)[0]
    polys = [l.polygon for l in pred.bboxes]
    positions = [str(l.position) for l in pred.bboxes]
    order_img = draw_polys_on_image(polys, img.copy(), labels=positions, label_font_size=20)
    return order_img, pred

def ocr(img, langs: List[str]) -> (Image.Image, OCRResult):
    replace_lang_with_code(langs)
    img_pred = run_ocr([img], [langs], det_model, det_processor, rec_model, rec_processor)[0]
    bboxes = [l.bbox for l in img_pred.text_lines]
    text = [l.text for l in img_pred.text_lines]
    rec_img = draw_text_on_image(bboxes, text, img.size, langs, has_math="_math" in langs)
    return rec_img, img_pred

def open_pdf(pdf_file):
    stream = io.BytesIO(pdf_file.getvalue())
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

def preprocess_image(image):
    # Convert the image to a numpy array if it's a PIL image
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binarization
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove noise
    denoised_image = cv2.fastNlMeansDenoising(binary_image, None, 30, 7, 21)

    # Sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(denoised_image, -1, kernel)

    # Resize the image
    resized_image = cv2.resize(sharpened_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    return resized_image

def extract_text_with_tesseract(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Apply Tesseract OCR
    custom_oem_psm_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(preprocessed_image, lang='ara+eng', config=custom_oem_psm_config)
    
    return text.splitlines()

def is_valid_date(date_str, date_format):
    try:
        date_obj = datetime.strptime(date_str, date_format)
        if 1900 <= date_obj.year <= 2100:
            return True
        return False
    except ValueError:
        return False

def filter_valid_dates(dates):
    valid_dates = []
    for date_str in dates:
        if '/' in date_str and is_valid_date(date_str, '%Y/%m/%d'):
            valid_dates.append(date_str)
        elif '-' in date_str and is_valid_date(date_str, '%d-%m-%Y'):
            valid_dates.append(date_str)
    return valid_dates

# Pattern to match "حبة" and any numbers
pattern_haba = re.compile(r'\bحبة\b')
pattern_numbers = re.compile(r'\d+|,\d+')

# Function to clean strings
def clean_string(s):
    s = re.sub(pattern_haba, '', s)  # Remove "حبة"
    s = re.sub(pattern_numbers, '', s)  # Remove numbers
    s = re.sub(r'\s{2,}', ' ', s)  # Remove extra spaces
    return s.strip()

def extract_invoice_data(ocr_result):
    # Joining the list of strings into a single text block
    ocr_text = '\n'.join(ocr_result)
    print("ocr_text", ocr_text)
    
    # Regex patterns
    amount_pattern = re.compile(r'\d{1,3}(?:,\d{3})*\.\d{2}')
    invoice_no_pattern = re.compile(r'رقم الفاتورة\s*(\d+)', re.UNICODE)
    date_pattern = re.compile(r'\d{4}/\d{2}/\d{2}')
    address_pattern = re.compile(r'Address\s*:\s*(.*?)\s*(?:Page|Invoice No|$)', re.DOTALL)
    phone_pattern = re.compile(r'\b\d{9,12}\b')
    salesman_pattern = re.compile(r'Salesman\s*:\s*(\S+)')
    notes_pattern = re.compile(r'Note\s*:\s*(.*?)\s*(?:\n|$)', re.DOTALL)
    digitsNum = re.compile(r'\b\d+(?![.,]\d{2}|\s*[/-])\b')

    # Extracting information
    discount = "0.00"
    amounts = amount_pattern.findall(ocr_text)
    invoice_no_match = invoice_no_pattern.search(ocr_text)
    dates = filter_valid_dates(date_pattern.findall(ocr_text)) 
    address = address_pattern.search(ocr_text)
    phone = phone_pattern.search(ocr_text)
    salesman = salesman_pattern.search(ocr_text)
    notes = notes_pattern.search(ocr_text)

    # Get invoice number
    invoice_no = invoice_no_match.group(1) if invoice_no_match else ''

    # Extract customer name
    customer_name = 'مؤسسة الأفكار المتنوعه لتقديم الوجبات'
    for line in ocr_result:
        if 'Cust.' in line:
            customer_name = line.split('Cust. : ')[-1].split(' القاريخ')[0].strip()
            break

    # Function to extract and map product details
    def extract_and_map_to_object(data, start_marker, end_marker):
        extracted_text = []
        is_extracting = False

        for index, item in enumerate(data):
            text = item
            if is_extracting and end_marker in text.strip():
                break

            if is_extracting:
                digits = digitsNum.findall(text)
                amounts = amount_pattern.findall(text)
                if len(amounts) > 0:
                    description = text
                    for match in amounts:
                        description = description.replace(match, '')
                    # Check the value at index 4 and add 1.00 if it's not 1.00
                    if len(amounts) > 4 and amounts[4] != '1.00':
                        amounts.insert(4, '1.00')

                    extracted_text.append({
                        "name": digits[-1] if digits else '',  # Assuming the last extracted digit is the name
                        "priceUnit": amounts[2] if len(amounts) > 4 else '',
                        "desc": clean_string(description),
                        "quantity": amounts[4] if len(amounts) > 4 else '',
                        "subtotal": amounts[0] if len(amounts) > 3 else '',
                    })
                         
            if start_marker in text.strip():
                is_extracting = True
        return extracted_text

    # Sample OCR text processing
    ocr_array = ocr_text.split("\n")
    sets_of_7_objects = extract_and_map_to_object(ocr_array, "Amount", "Delivery")

    print("invoice_no", invoice_no)
    print("customer_name", customer_name)

    # Compile extracted data into a dictionary
    invoice_data = {
        "results": [
            {
                "extracted details": {
                    "Address (العنوان)": address.group(1).strip() if address else '',
                    "Phone (هاتف)": phone.group(0) if phone else '',
                    "Customer (العميل)": customer_name,
                    "Sale Person (اسم البائع)": salesman.group(1) if salesman else '',
                    "Invoice No. (رقم الفاتورة)": invoice_no,
                    "Discount (مجموع الخصم)": discount,
                    "Due Date (تاريخ الاستحقاق)": dates[0] if len(dates) > 1 else '',
                    "Invoice Date (تاريخ الفاتورة)": dates[1] if len(dates) > 0 else '',
                    "Notes": notes.group(1).strip() if notes else '',
                    "Products Info": [
                        {
                            "Nature of Goods or service (تفاصيل السلع أو الخدمات)": obj["name"],
                            "Price Unit (سعر الوحدة)": obj["priceUnit"],
                            "Description وصف": obj["desc"],
                            "Quantity (الكمية)": obj["quantity"],
                            "Subtotal (Including VAT) (الاجمالي شامل الضريبة)": obj["subtotal"],
                        }
                        for obj in sets_of_7_objects
                    ]
                }
            }
        ]
    }
    # process_invoices(invoice_data)
    return invoice_data

def process_file(in_file, languages):
    filetype = in_file.type
    if "pdf" in filetype:
        num_pages = get_pdf_page_count(in_file)
        pil_image = get_page_image(in_file, 1)  # Process only the first page for simplicity
    else:
        pil_image = Image.open(in_file).convert("RGB")
    
    results = {
        "filename": in_file.name,
        "image": pil_image,
        "text_detection": None,
        "layout_detection": None,
        "ocr": None,
        "order_detection": None
    }
    
    if text_det:
        results["text_detection"] = text_detection(pil_image)
    
    if layout_det:
        results["layout_detection"] = layout_detection(pil_image)
    
    if text_rec:
        results["ocr"] = ocr(pil_image, languages)
    
    if order_det:
        results["order_detection"] = order_detection(pil_image)
    
    return results

st.set_page_config(layout="wide")

det_model, det_processor = load_det_cached()
rec_model, rec_processor = load_rec_cached()
layout_model, layout_processor = load_layout_cached()
order_model, order_processor = load_order_cached()

st.markdown("""
# Arabic Receipts OCR Demo

This app will let you try a multilingual OCR model. It supports text detection + layout analysis in any language, and text recognition in several languages.

Notes:
- This works best on documents with printed text.
- Preprocessing the image (e.g. increasing contrast) can improve results.
- If OCR doesn't work, try changing the resolution of your image (increase if below 2048px width, otherwise decrease).
- You can now upload multiple files and process them all at once.

""")

# # Load JSON data from a file
# with open('data1.json', 'r', encoding='utf-8') as file:
#     data = json.load(file)

# # Streamlit application
# st.title("OCR JSON Viewer")

# with st.expander("OCR JSON"):
#     text_values = [item["text"] for item in data["text_lines"]]
#     st.json(text_values)

in_files = st.sidebar.file_uploader("PDF file(s) or image(s):", type=["pdf", "png", "jpg", "jpeg", "gif", "webp"], accept_multiple_files=True)
languages = st.sidebar.multiselect("Languages", sorted(list(CODE_TO_LANGUAGE.values())), default=["English"], max_selections=4)

if not in_files:
    st.stop()

text_det = st.sidebar.button("Run Text Detection")
text_rec = st.sidebar.button("Run OCR")
layout_det = st.sidebar.button("Run Layout Analysis")
order_det = st.sidebar.button("Run Reading Order")


if text_det or text_rec or layout_det or order_det:
    all_results = []
    for in_file in in_files:
        results = process_file(in_file, languages)
        all_results.append(results)
    
    for index, result in enumerate(all_results):
        st.subheader(f"Results for {result['filename']} (File {index + 1} of {len(all_results)})")
        col1, col2 = st.columns([.5, .5])
        
        with col2:
            st.image(result['image'], caption="Uploaded Image", use_column_width=True)
        
        with col1:
            if result['text_detection']:
                det_img, pred = result['text_detection']
                st.image(det_img, caption="Detected Text", use_column_width=True)
                with st.expander("Text Detection JSON"):
                    st.json(pred.model_dump(exclude=["heatmap", "affinity_map"]))
            
            if result['layout_detection']:
                layout_img, pred = result['layout_detection']
                st.image(layout_img, caption="Detected Layout", use_column_width=True)
                with st.expander("Layout Detection JSON"):
                    st.json(pred.model_dump(exclude=["segmentation_map"]))
            
            if result['ocr']:
                rec_img, pred = result['ocr']
                recognized_text_tesseract = extract_text_with_tesseract(rec_img)
                invoice_data = ""
                ans = " ".join(recognized_text_tesseract)
                if "Gulf Fiberglass Fac & polyethylene" in ans:
                    print("Gulf Fiberglass Fac & polyethylene")
                    print("recognized_text_tesseract", recognized_text_tesseract)
                    invoice_data = extract_invoice_data_2(recognized_text_tesseract)
                else:
                    print("recognized_text_tesseract", recognized_text_tesseract)
                    # invoice_data = extract_invoice_data(recognized_text_tesseract)
                
                st.image(rec_img, caption="OCR Result", use_column_width=True)
                with st.expander("OCR JSON"):
                    st.json(invoice_data)
                with st.expander("Text Lines (for debugging)"):
                    st.text("\n".join([p.text for p in pred.text_lines if p.text.strip()]))
            
            if result['order_detection']:
                order_img, pred = result['order_detection']
                st.image(order_img, caption="Reading Order", use_column_width=True)
                with st.expander("Reading Order JSON"):
                    st.json(pred.model_dump())
        
        st.markdown("---")