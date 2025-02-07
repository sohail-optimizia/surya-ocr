import re
from datetime import datetime
from xmlrpc import client as xmlrpclib
from decimal import Decimal, InvalidOperation

# Regex patterns
amount_pattern = re.compile(r'\b\d{1,3}(?:,\d{3})+(?:\.\d{2})?\b|\b\d+\.\d{2}\b')
invoice_no_pattern = re.compile(r'رقم الفاتورة\s*(\d+)', re.UNICODE)
date_pattern = re.compile(r'\b\d{2}/\d{2}/\d{4}\b')
address_pattern = re.compile(r'Address\s*:\s*(.*?)\s*(?:Page|Invoice No|$)', re.DOTALL)
phone_pattern = re.compile(r'\b\d{9,12}\b')
salesman_pattern = re.compile(r'Salesman\s*:\s*(\S+)')
notes_pattern = re.compile(r'Note\s*:\s*(.*?)\s*(?:\n|$)', re.DOTALL)
digitsNum = re.compile(r'\b\d+\b')
email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

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
            # Convert string to datetime object using the correct format
            invoice_date_due = datetime.strptime(invoice_date_due, '%d/%m/%Y').strftime('%Y-%m-%d')
            print("Formatted date:", invoice_date_due)
        except ValueError:
            # Handle invalid date format gracefully
            invoice_date_due = False
            print("Invalid date format")
        
        if invoice_date:
            try:
                # Convert string to datetime object using the correct format
                invoice_date = datetime.strptime(invoice_date, '%d/%m/%Y').strftime('%Y-%m-%d')
                print("Formatted date:", invoice_date)
            except ValueError:
                # Handle invalid date format gracefully
                invoice_date = False
                print("Invalid date format")

        # Extract products
        for product_info in extracted_details.get('Products Info', []):
            product_name = product_info.get('Nature of Goods or service (تفاصيل السلع أو الخدمات)', '')
            product_quantity = product_info.get('Quantity (الكمية)', '')
            
            # Handle list price with commas and dots
            list_price_str = product_info.get('Price Unit (\u0633\u0639\u0631 \u0627\u0644\u0648\u062d\u062f\u0629)', '0.0')
            list_price_str = list_price_str.replace(',', '')  # Remove commas
            try:
                list_price = float(list_price_str)  # Convert to float
            except ValueError:
                list_price = 0.0  # Default to 0.0 if conversion fails

            product_data = {
                "name": product_name,
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

        country_seller_id = models.execute_kw(ODOO_DB, uid, ODOO_PASSWORD, 'res.country', 'search', [[('name', '=', "Saudi Arabia")]], {'limit': 1})
        
        phone_seller = extracted_details["Customer Phone (هاتف)"]
        seller_name = extracted_details['Customer (العميل)']
        seller_vat = extracted_details["Customer Vat لرقم الضريبي"]

        seller_data = {
            'name': seller_name, 
            "is_company": True,
            "street": "",
            "zip": "",
            "city": "",
            "country_id": country_seller_id[0],
            "vat": seller_vat ,
            "phone": phone_seller
         }

        customer_name = extracted_details["Sale Person (اسم البائع)"]
        customer_address = extracted_details["Address (العنوان)"]
        customer_email = extracted_details["Seller Email ي ا"]
        customer_phone = extracted_details["Seller Phone (هاتف)"]
        customer_vat = extracted_details["Seller Vat لرقم الضريبي"]

        customer_data = {
            'name': customer_name, 
            "is_company": True,
            "street": customer_address,
            "zip": "",
            "city": "",
            "country_id": country_seller_id[0],
            "vat": customer_vat,
            "phone": customer_phone,
            "email": customer_email,
            "website": ""
        }

        # Check if the seller exists in Odoo
        seller_id = models.execute_kw(ODOO_DB, uid, ODOO_PASSWORD, 'res.partner', 'search', [[('name', '=', seller_name)]], {'limit': 1})
        if not seller_id:
            # Create seller if not exists
            seller_id = models.execute_kw(ODOO_DB, uid, ODOO_PASSWORD, 'res.partner', 'create', [seller_data])
        

        # Check if the seller exists in Odoo
        customer_id = models.execute_kw(ODOO_DB, uid, ODOO_PASSWORD, 'res.partner', 'search', [[('name', '=', customer_name)]], {'limit': 1})
        if not customer_id:
            customer_id = models.execute_kw(ODOO_DB, uid, ODOO_PASSWORD, 'res.partner', 'create', [customer_data])
            bank_name_to_find = 'نك ساب'
            # Search for the bank name in res.bank model
            bank_ids = models.execute_kw(ODOO_DB, uid, ODOO_PASSWORD, 'res.bank', 'search', [[('name', '=', bank_name_to_find)]])
            # Create seller if not exists
            # Prepare data to create a bank account
            bank_account_data = {
                'acc_number': 'SA6045000000044058006001',  # Replace with actual account number
                'partner_id': customer_id,
                'bank_name': bank_ids[0],  # Replace with actual bank name
                #'currency_id': 1,   Replace with actual currency ID
                # Add other fields as needed (e.g., iban, bank_bic, ...)
            }
            # Create the bank account record
            bank_account_id = models.execute_kw(ODOO_DB, uid, ODOO_PASSWORD, 'res.partner.bank', 'create', [bank_account_data])

            if bank_account_id:
                print(f"Bank account created successfully with ID: {bank_account_id}")
            else:
                print("Failed to create bank account.")

        # Prepare invoice data to create in Odoo
        invoice_data = {
            "name": str(name),
            "display_name": str(name),
            "invoice_user_id": False,
            "partner_id": customer_id if isinstance(customer_id, int) else customer_id[0],
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


def is_valid_date(date_str, date_format):
    try:
        datetime.strptime(date_str, date_format)
        return True
    except ValueError:
        return False

def filter_valid_dates(dates):
    valid_dates = []
    for date_str in dates:
        if '/' in date_str and is_valid_date(date_str, '%d/%m/%Y'):
            valid_dates.append(date_str)
        elif '-' in date_str and is_valid_date(date_str, '%d-%m-%Y'):
            valid_dates.append(date_str)
    return valid_dates

def extract_specific_numbers(text):
    # This regex will match numbers except for those preceded or followed by "©" or preceded by "ولا"
    extras = digitsNum.findall(text)
    return extras

def extract_invoice_data_2(ocr_result):
    # Joining the list of strings into a single text block
    ocr_text = '\n'.join(ocr_result)   

    # Extracting information
    discount = "0.00"
    amounts = amount_pattern.findall(ocr_text)
    invoice_no_match = invoice_no_pattern.search(ocr_text)
    dates = filter_valid_dates(date_pattern.findall(ocr_text)) 
    notes = notes_pattern.search(ocr_text)
    # Get invoice number
    invoice_no = invoice_no_match.group(1) if invoice_no_match else ''
    # Extract customer name

    salesman = salesman_pattern.search(ocr_text)
    seller_address = ""
    seller_email = ""
    seller_phone = ""
    seller_vat = ""
    ## Customer data
    customer_vat = ""
    customer_name = 'مزمسمة الافكار المنتوعة لتقديم الوجبات'
    customer_phone = ""
    for index, line in enumerate(ocr_result):
        if index == 1: salesman = line
        if index == 2: seller_address = line
        if "ناريخ استحقاق القاتورة" in line:
           customer_vat = extract_specific_numbers(line)[0]
        if "Phone" in line:
            customer_phone = extract_specific_numbers(line)[0]
        if "Invoice No" in line:
            numbers = extract_specific_numbers(line)
            invoice_no = numbers[len(numbers) - 1]
        if "Email" in line:
            seller_email = re.findall(email_pattern, line)[0]
        if "رقم القواصل" in line:
            seller_phone = extract_specific_numbers(line)[0]
        if "لرقم الضريبي" in line and seller_vat == "":
            seller_vat = extract_specific_numbers(line)[0]

    # Function to extract and map product details
    def extract_and_map_to_object(data, start_marker, end_marker):
        extracted_text = []
        is_extracting = False

        for index, item in enumerate(data):
            text = item
            if is_extracting and end_marker in text.strip():
                break

            if is_extracting:
                amounts = amount_pattern.findall(text)
                if len(amounts) > 0:
                    # Extract matches and remove them from the original string
                    name = text
                    for match in amounts:
                        name = name.replace(match, '')
                    name = ' '.join(name.split())
                    # Check the value at index 4 and add 1.00 if it's not 1.00
                    if len(amounts) > 4 and amounts[4] != '1.00':
                        amounts.insert(4, '1.00')

                    extracted_text.append({
                        "name": name,  # Assuming the last extracted digit is the name
                        "priceUnit": amounts[0] if len(amounts) > 4 else '',
                        "quantity": amounts[1] if len(amounts) > 4 else '',
                        "subtotal": amounts[6] if len(amounts) > 3 else '',
                    })
                         
            if start_marker in text.strip():
                is_extracting = True
        return extracted_text

    # Sample OCR text processing
    ocr_array = ocr_text.split("\n")
    sets_of_7_objects = extract_and_map_to_object(ocr_array, "الضريية يلع", "أممم البنك")

    # Compile extracted data into a dictionary
    invoice_data = {
        "results": [
            {
                "extracted details": {
                    "Address (العنوان)": seller_address,
                    "Customer Phone (هاتف)": customer_phone,
                    "Customer (العميل)": customer_name,
                    "Customer Vat لرقم الضريبي": customer_vat,
                    "Seller Email ي ا": seller_email,
                    "Seller Phone (هاتف)":seller_phone,
                    "Seller Vat لرقم الضريبي": seller_vat,
                    "Sale Person (اسم البائع)": salesman if salesman else '',
                    "Invoice No. (رقم الفاتورة)": invoice_no,
                    "Discount (مجموع الخصم)": discount,
                    "Due Date (تاريخ الاستحقاق)": dates[0] if len(dates) > 1 else '',
                    "Invoice Date (تاريخ الفاتورة)": dates[1] if len(dates) > 0 else '',
                    "Notes": notes.group(1).strip() if notes else '',
                    "Products Info": [
                        {
                            "Nature of Goods or service (تفاصيل السلع أو الخدمات)": obj["name"],
                            "Price Unit (سعر الوحدة)": obj["priceUnit"],
                            "Quantity (الكمية)": obj["quantity"],
                            "Subtotal (Including VAT) (الاجمالي شامل الضريبة)": obj["subtotal"],
                        }
                        for obj in sets_of_7_objects
                    ]
                }
            }
        ]
    }
    process_invoices(invoice_data)
    return invoice_data

# ocr_result2 = [
#   "مصنع الخليج للفير جلاس و البولجا أبشبلين مصنع الخليج للفيبرجلاس والبولي ايثلين للصناعة",
#   "‎GULF FIBERGLASS FACTORY & POLYETHLENE‏ \"",
#   "7, الدمام, ‎Dallah Industrial, omar ibn alkhattab‏ ,",
#   "البريد الإلكتروني ‎Email gulf_fiberglass@hotmail.com‏",
#   "رقم القواصل 0509334644 ‎Contact‏",
#   "‎Number‏",
#   "‎Tax Invoice‏ الرقم الذ ‎VAT‏",
#   "‏لرقم الضريبي 300510005700003",
#   "‎Number arr‏",
#   "فاتورة ضريبية",
#   "‎Company 45,5‏ مزمسمة الافكار المنتوعة لتقديم الوجبات رقم الفاتورة 988 ‎Invoice No‏",
#   "رقم التواصل ‎Phone‏ 0506922244 ناريخ إصدار الفاتورة 19/12/2022 ‎Invoice Issue Date‏",
#   "الرقم الضريبي ‎Vat Number‏ 300465742800003 ناريخ استحقاق القاتورة 19/12/2022 ‎Invoice Due Date‏",
#   "‎Items Subtotal Tax Amount Tax Rate Taxable Amount Quantity Unit Price Item Name‏",
#   "معر الموحدة لكمنية مبلع الحاضيع للضر ببة سبة الضريية يلع الضربية المجموع شامل الضربية",
#   "خز ان الخطيب سعة 2000 لنر ضمان 15 سنه 869.57 1.00 869.57 15.00 130.43 1,000",
#   "أممم البنك 213006 881016 المجموع الفرعي",
#   "‎1g:‏ 869.57",
#   "‎Total (Excluding VAT)‏",
#   "بنك ساب",
#   "الإجمالي غير شامل الضربية",
#   "‎IBAN Number ,;iy 31 a3;‏ . 869.57",
#   "‎Total Taxable Amounts {Excl. VAT}‏",
#   "56604500000001 : : 0",
#   "مجموع ضريبة القيمة المضافة",
#   "130.43",
#   "‎Total VAT Name aw2l‏",
#   "‎Gulf Fiberglass Fac & polyethylene‏ إجمالي المبلغ المستحق",
#   "‎SAR 1,000 Total Amount Due‏",
#   "الملاحظات 10165",
#   "ضمان على خز انات الفايبر 25 سنة ضمان على خز انات البولي",
#   "ايثلين 15 سنة",
#   "| مطوّر من قيل وصول | انا05/لا 87 010/5850م",
#   ""
# ]

# import json
# print(json.dumps(invoice_data, ensure_ascii=False, indent=4))