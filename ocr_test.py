import argparse
import os
import io
import json
from typing import List
import pandas as pd
import requests
import pypdfium2
import streamlit as st
from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection
from surya.model.detection.segformer import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from surya.model.ordering.processor import load_processor as load_order_processor
from surya.model.ordering.model import load_model as load_order_model
from surya.ocr import run_ocr
from surya.postprocessing.text import draw_text_on_image
from PIL import Image
from surya.languages import CODE_TO_LANGUAGE
from surya.input.langs import replace_lang_with_code
from surya.schema import OCRResult
import xmlrpc.client as xmlrpclib
from datetime import datetime
from google import genai
import openai

parser = argparse.ArgumentParser(description="Run OCR on an image or PDF.")
parser.add_argument("--math", action="store_true", help="Use math model for detection", default=False)

try:
    args = parser.parse_args()
except SystemExit as e:
    print(f"Error parsing arguments: {e}")
    os._exit(e.code)

@st.cache_resource()
def load_rec_cached():
    return load_rec_model(), load_rec_processor()

@st.cache_resource()
def load_det_cached():
    return load_det_model(), load_det_processor()

rec_model, rec_processor = load_rec_cached()
det_model, det_processor = load_det_cached()

def ocr(img, langs: List[str]) -> OCRResult:
    replace_lang_with_code(langs)
    img_pred = run_ocr([img], [langs], det_model=det_model, det_processor=det_processor, rec_model=rec_model, rec_processor=rec_processor)[0]
    return img_pred

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

def is_text_based_pdf(pdf_file) -> bool:
    doc = open_pdf(pdf_file)
    for page_num in range(min(3, len(doc))):  # Check the first 3 pages
        page = doc[page_num]
        text = page.get_textpage().get_text_range()
        if text.strip():
            return True
    return False

def extract_text_from_pdf(pdf_file):
    doc = open_pdf(pdf_file)
    extracted_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_textpage().get_text_range()
        extracted_text.append(text)
    return "\n".join(extracted_text)

def process_file(file_path, languages):
    with open(file_path, 'rb') as f:
        in_file = io.BytesIO(f.read())
    filetype = file_path.lower().split('.')[-1]
    if filetype == "pdf":
        num_pages = get_pdf_page_count(in_file)
        pil_image = get_page_image(in_file, 1)  # Process only the first page for simplicity
        is_text_based = is_text_based_pdf(in_file)
        pdf_type = "Text-Based" if is_text_based else "Scanned"
        st.write(f"Detected PDF type: {pdf_type}")
        if is_text_based:
            text = extract_text_from_pdf(in_file)
            results = {
                "filename": os.path.basename(file_path),
                "image": pil_image,
                "text": text,
                "pdf_type": pdf_type,
                "ocr": None,
            }
        else:
            text = None
            ocr_result = ocr(pil_image, languages)
            results = {
                "filename": os.path.basename(file_path),
                "image": pil_image,
                "text": None,
                "pdf_type": pdf_type,
                "ocr": ocr_result,
            }
    else:
        pil_image = Image.open(in_file).convert("RGB")
        pdf_type = "Image"
        text = None
        ocr_result = ocr(pil_image, languages)
        results = {
            "filename": os.path.basename(file_path),
            "image": pil_image,
            "text": None,
            "pdf_type": pdf_type,
            "ocr": ocr_result,
        }

    return results

def ocr_results_to_csv(results):
    """Convert OCR results to CSV format with structured data."""
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
                    # If bbox is not in the expected format, log or handle the issue
                    print(f"Unexpected bbox format: {bbox}")
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

def ocr_results_to_json(results):
    """Convert OCR results to JSON format with structured data."""
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
                    # If bbox is not in the expected format, log or handle the issue
                    print(f"Unexpected bbox format: {bbox}")
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

st.title("Surya OCR Application")

languages = st.sidebar.multiselect(
    "Select OCR languages",
    options=list(CODE_TO_LANGUAGE.keys()),
    default=["en"]
)

# Odoo configurations
ODOO_URL = 'https://students8.odoo.com/'
ODOO_DB = 'students8'
ODOO_USERNAME = 'malik.faizan@optimizia.co'
ODOO_PASSWORD = 'Cancel123@@'
ODOO_MODEL = 'account.move'  # Model name for invoices

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

def parse_date(date_str):
    possible_formats = ['%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y']
    for fmt in possible_formats:
        try:
            return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    return None  # If all formats fail, return None

def dump_to_odoo(organized_data):
    models = xmlrpclib.ServerProxy('{}/xmlrpc/2/object'.format(ODOO_URL), allow_none=True)
    print("organized_data", organized_data['invoice_header'])
    invoice_header = organized_data['invoice_header']
    customer = invoice_header['customer']
    supplier = invoice_header['supplier']
    items = organized_data['items']
    notes = invoice_header.get('notes', '')

    required_fields = ['invoice_number', 'invoice_date', 'customer', 'supplier', 'items']
    for field in required_fields:
        if not invoice_header.get(field) and not organized_data.get(field):
            print(f"Error: Missing required field '{field}'")
            return

    if not customer.get('name'):
        print("Error: Missing customer name.")
        return

    if not supplier.get('name') or not supplier.get('vat_number'):
        print("Error: Missing supplier name or VAT number.")
        return

    buyer_name = customer['name']
    buyer_phone = customer['phone']
    buyer_data = {
        'name': buyer_name,
        "is_company": True,
        "phone": buyer_phone,
        "vat": customer.get('vat_number', '')
    }

    buyer_id = models.execute_kw(ODOO_DB, uid, ODOO_PASSWORD, 'res.partner', 'search', [[('name', '=', buyer_name)]], {'limit': 1})
    if not buyer_id:
        buyer_id = models.execute_kw(ODOO_DB, uid, ODOO_PASSWORD, 'res.partner', 'create', [buyer_data])
    else:
        buyer_id = buyer_id[0]

    supplier_name = supplier['name']
    supplier_phone = supplier['phone']
    supplier_data = {
        'name': supplier_name,
        "is_company": True,
        "phone": supplier_phone,
        "vat": supplier.get('vat_number', '')
    }

    supplier_id = models.execute_kw(ODOO_DB, uid, ODOO_PASSWORD, 'res.partner', 'search', [[('name', '=', supplier_name)]], {'limit': 1})
    if not supplier_id:
        supplier_id = models.execute_kw(ODOO_DB, uid, ODOO_PASSWORD, 'res.partner', 'create', [supplier_data])
    else:
        supplier_id = supplier_id[0]

    invoice_lines = []
    for item in items:
        product_name = item.get('name')
        product_quantity = item.get('quantity', 1)  # Default to 1 if missing
        unit_price = item.get('unit_price')

        if not product_name or not unit_price:
            print(f"Warning: Incomplete item information: {item}. Skipping...")
            continue

        # Ensure product_quantity is a valid number
        try:
            product_quantity = float(product_quantity) if product_quantity else 1  # Ensure it's a number
        except ValueError:
            product_quantity = 1  # Default to 1 if conversion fails

        # Remove commas and convert unit_price to float
        try:
            if isinstance(unit_price, str):
                unit_price = float(unit_price.replace(',', ''))
            elif isinstance(unit_price, (int, float)):
                # If it's already a float or int, no need to replace commas
                unit_price = float(unit_price)
            else:
                raise ValueError("unit_price must be a string, int, or float")        
        except ValueError:
                    print(f"Error: Unable to convert unit_price to float for item: {product_name}. Skipping...")
                    continue

        product_data = {
            "name": product_name,
            "list_price": unit_price
        }

        product_id = models.execute_kw(ODOO_DB, uid, ODOO_PASSWORD, 'product.template', 'search', [[('name', '=', product_name)]], {'limit': 1})

        if not product_id:
            product_id = models.execute_kw(ODOO_DB, uid, ODOO_PASSWORD, 'product.template', 'create', [product_data])
        else:
            product_id = product_id[0]

        invoice_lines.append([0, 0, {
            "product_id": product_id,
            "quantity": float(product_quantity),
        }])

    try:
        # invoice_date = datetime.strptime(invoice_header['invoice_date'], '%d/%m/%Y').strftime('%Y-%m-%d')
        invoice_date = parse_date(invoice_header.get('invoice_date'))
    except ValueError:
        print("Error: Invalid invoice date format.")
        return
    
    due_date = invoice_header.get('due_date')
    if due_date:
        try:
            due_date = datetime.strptime(due_date, '%d/%m/%Y').strftime('%Y-%m-%d')
        except ValueError:
            print("Error: Invalid due date format.")
            return
    else:
        due_date = None

    # Prepare the invoice data dictionary
    invoice_data = {
        "name": invoice_header['invoice_number'],
        "partner_id": buyer_id,
        "move_type": "out_invoice",
        "invoice_date": invoice_date,
        "invoice_line_ids": invoice_lines,
        "narration": notes
    }

    # Only include the due date if it's provided and valid
    if due_date:
        invoice_data["invoice_date_due"] = due_date

    print("invoice_data", invoice_data)

    try:
        invoice_id = models.execute_kw(ODOO_DB, uid, ODOO_PASSWORD, ODOO_MODEL, 'create', [invoice_data])
        print(f"Invoice '{invoice_header['invoice_number']}' created with ID: {invoice_id}")
    except Exception as e:
        print(f"Failed to create invoice '{invoice_header['invoice_number']}': {str(e)}")

openai.api_key = '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'

# def organize_invoice_data(invoice_data):
#     print("invoice_data", invoice_data)
#     json_template = """{
#         "organized_data": {
#             "invoice_header": {
#                 "bank": {
#                     "iban": "",
#                     "name": ""
#                 },
#                 "customer": {
#                     "name": "",
#                     "phone": "",
#                     "vat_number": ""
#                 },
#                 "due_date": "",
#                 "invoice_date": "",
#                 "invoice_number": "",
#                 "notes": "",
#                 "supplier": {
#                     "address": "",
#                     "email": "",
#                     "name": "",
#                     "phone": "",
#                     "vat_number": ""
#                 }
#             },
#             "items": [
#                 {
#                     "name": "",
#                     "quantity": "",
#                     "unit_price": ""
#                 }
#             ],
#             "totals": {
#                 "subtotal": "",
#                 "total_amount_due": "",
#                 "total_taxable_amount": "",
#                 "total_vat": ""
#             }
#         }
#     }"""

#     prompt = f"""
#     Organize the following invoice data into a structured format and return only json:

#     Invoice Data: {json.dumps(invoice_data)}

#     """

#     # Provide the organized data in JSON format follow this structure:
#     # {json_template}

#     response = openai.ChatCompletion.create(
#     model="gpt-4o-mini", #gpt-3.5-turbo
#     messages=[
#         # {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": prompt}
#     ]
#     )
#     print(response['choices'][0]['message']['content'])

#     organized_data = response['choices'][0]['message']['content']
#     return json.loads(organized_data)

def organize_invoice_data(invoice_data):
    print("invoice_data", invoice_data)
    
    json_template = """{
        "organized_data": {
            "invoice_header": {
                "bank": {
                    "iban": "",
                    "name": ""
                },
                "customer": {
                    "name": "",
                    "phone": "",
                    "vat_number": ""
                },
                "due_date": "",
                "invoice_date": "",
                "invoice_number": "",
                "notes": "",
                "supplier": {
                    "address": "",
                    "email": "",
                    "name": "",
                    "phone": "",
                    "vat_number": ""
                }
            },
            "items": [
                {
                    "name": "",
                    "quantity": "",
                    "unit_price": ""
                }
            ],
            "totals": {
                "subtotal": "",
                "total_amount_due": "",
                "total_taxable_amount": "",
                "total_vat": ""
            }
        }
    }"""

    prompt = f"""
    Organize the following invoice data into a structured format like this {json_template} also check for data types and return only json do not add ''' and json:
    
    Invoice Data: {json.dumps(invoice_data)}

    """  
    
    # Initialize the Gemini client
    client = genai.Client(api_key="AIzaSyCMJEALd5cy9KPlwjBgHN5u1Lll8nWgjfc")
    
    # Request to Gemini API for organizing the invoice data
    response = client.models.generate_content(
        model="gemini-2.0-flash",  # Use the appropriate model
        contents=prompt  # Send the prompt to the API
    )

    # Clean the response text by stripping unwanted spaces, newlines, and "```json"
    response_text = response.text.strip()

    # Remove the unwanted "```json" and other unwanted characters (if present)
    if response_text.startswith("```json"):
        response_text = response_text[len("```json"):].strip()  # Remove the "```json" part
    if response_text.endswith("```"):
        response_text = response_text[:-3].strip()  # Remove the ending "```" part
    
    # Print the cleaned raw response for debugging
    print("Cleaned raw response:", response_text)
    
    # Try to parse the cleaned response text into JSON
    try:
        organized_data = json.loads(response_text)  # Now it should parse correctly
        return organized_data
    except json.JSONDecodeError as e:
        print(f"Failed to parse response as JSON: {e}")
        return None

# Streamlit code to add a button to set the folder path and process files
if st.sidebar.button("Use Default Folder Path"):
    folder_path = os.path.join(os.getcwd(), "./invoices")
    
    # List files with extensions .pdf, .jpg, .jpeg, .png
    files = [
        os.path.join(folder_path, f) 
        for f in os.listdir(folder_path) 
        if f.lower().endswith((".pdf", ".jpg", ".jpeg", ".png"))
    ]
    
    # Process each file
    results = [process_file(file, languages) for file in files]

    # Display results
    for result in results:
        st.header(f"File: {result['filename']}")
        
        if result['pdf_type'] == "Text-Based":            
            # Organize the extracted text using Gemini API
            gemini_result = organize_invoice_data(result['text'])
            if "error" in gemini_result:
                st.error(f"Error: {gemini_result['error']}")
            else:
                st.subheader("Organized Data")
                dump_to_odoo(gemini_result['organized_data'])
                st.json(gemini_result['organized_data'])
        else:
            if result.get("ocr"):
                st.subheader("OCR")
                for line in result["ocr"].text_lines:
                    st.text(line.text)
                
                # Combine OCR text for organizing
                combined_ocr_text = "\n".join(line.text for line in result["ocr"].text_lines)
                # Organize the OCR text using Gemini API
                gemini_result = organize_invoice_data(combined_ocr_text)
                if "error" in gemini_result:
                    st.error(f"Error: {gemini_result['error']}")
                else:
                    st.subheader("Organized Data")
                    # dump_to_odoo(gemini_result['organized_data'])
                    st.json(gemini_result['organized_data'])






















# def translateContent(data):
    
#     response = openai.ChatCompletion.create(
#     model="gpt-4o", #gpt-3.5-turbo
#     messages=[
#         # {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": prompt}
#     ]
#     )
#     print(response['choices'][0]['message']['content'])

#     translated_data = response['choices'][0]['message']['content']
#     return json.loads(translated_data)
    