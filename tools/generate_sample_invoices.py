#!/usr/bin/env python
"""
Generate sample invoice documents for testing Document AI processor.
This script creates PDF invoices with various layouts and content
for training and testing the Document AI invoice processor.
"""

import os
import random
from datetime import datetime, timedelta
import argparse
from typing import Dict, Any
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT

# Company data for sample invoices
VENDORS = [
    {"name": "Acme Supplies Ltd", "address": "123 Business Ave, San Francisco, CA 94103", "tax_id": "12-3456789"},
    {"name": "Tech Solutions Inc", "address": "456 Innovation Blvd, Austin, TX 78701", "tax_id": "98-7654321"},
    {"name": "Global Logistics Corp", "address": "789 Shipping Lane, Seattle, WA 98101", "tax_id": "45-6789012"},
    {"name": "Office Smart", "address": "321 Corporate Dr, Chicago, IL 60601", "tax_id": "34-5678901"},
    {"name": "Green Energy Co", "address": "555 Eco Way, Portland, OR 97201", "tax_id": "56-7890123"}
]

CUSTOMERS = [
    {"name": "XYZ Corporation", "address": "100 Main St, New York, NY 10001", "tax_id": "11-2233445"},
    {"name": "Innovative Startups LLC", "address": "200 Creative Ave, Boston, MA 02110", "tax_id": "22-3344556"},
    {"name": "Big Enterprise Group", "address": "300 Corporate Blvd, Los Angeles, CA 90001", "tax_id": "33-4455667"},
    {"name": "Small Business Partners", "address": "400 Commerce St, Miami, FL 33101", "tax_id": "44-5566778"},
    {"name": "Research Institute", "address": "500 Science Park, Denver, CO 80201", "tax_id": "55-6677889"}
]

PRODUCTS = [
    {"name": "Office Chair - Standard", "unit_price": 129.99},
    {"name": "Office Chair - Executive", "unit_price": 249.99},
    {"name": "Desk - Regular", "unit_price": 199.99},
    {"name": "Desk - Standing", "unit_price": 399.99},
    {"name": "Laptop - Basic", "unit_price": 699.99},
    {"name": "Laptop - Premium", "unit_price": 1299.99},
    {"name": "Monitor - 24\"", "unit_price": 179.99},
    {"name": "Monitor - 32\" 4K", "unit_price": 349.99},
    {"name": "Printer - Laser", "unit_price": 249.99},
    {"name": "Printer - Color", "unit_price": 349.99},
    {"name": "Software License - Basic", "unit_price": 99.99},
    {"name": "Software License - Premium", "unit_price": 199.99},
    {"name": "Cloud Storage (1TB)", "unit_price": 9.99},
    {"name": "Server Hosting (Monthly)", "unit_price": 49.99},
    {"name": "Tech Support (Hourly)", "unit_price": 79.99}
]

PAYMENT_TERMS = [
    "Net 30",
    "Net 45",
    "Net 60",
    "2% 10, Net 30",
    "Due on Receipt"
]

def generate_invoice_data() -> Dict[str, Any]:
    """Generate random invoice data."""
    # Select vendor and customer
    vendor = random.choice(VENDORS)
    customer = random.choice(CUSTOMERS)
    
    # Generate invoice number
    invoice_number = f"INV-{random.randint(10000, 99999)}"
    
    # Generate dates
    today = datetime.now()
    issue_date = today - timedelta(days=random.randint(1, 30))
    due_date = issue_date + timedelta(days=random.randint(15, 60))
    
    # Generate line items
    num_items = random.randint(2, 8)
    line_items = []
    subtotal = 0
    
    for _ in range(num_items):
        product = random.choice(PRODUCTS)
        quantity = random.randint(1, 5)
        unit_price = product["unit_price"]
        amount = quantity * unit_price
        subtotal += amount
        
        line_items.append({
            "description": product["name"],
            "quantity": quantity,
            "unit_price": unit_price,
            "amount": amount
        })
    
    # Calculate tax
    tax_rate = random.choice([0.05, 0.06, 0.07, 0.08, 0.09, 0.0])
    tax_amount = round(subtotal * tax_rate, 2)
    
    # Calculate total
    total_amount = subtotal + tax_amount
    
    return {
        "vendor": vendor,
        "customer": customer,
        "invoice_number": invoice_number,
        "issue_date": issue_date.strftime("%Y-%m-%d"),
        "due_date": due_date.strftime("%Y-%m-%d"),
        "line_items": line_items,
        "subtotal": subtotal,
        "tax_rate": tax_rate,
        "tax_amount": tax_amount,
        "total_amount": total_amount,
        "payment_terms": random.choice(PAYMENT_TERMS),
        "currency": "USD"
    }

def create_invoice_pdf(invoice_data: Dict[str, Any], output_path: str) -> None:
    """Create PDF invoice from data."""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Define custom styles
    try:
        # Custom Title style
        styles.add(ParagraphStyle(
            name='InvoiceTitle',
            fontSize=16,
            alignment=TA_CENTER,
            spaceAfter=12
        ))
    except KeyError:
        # Style already exists, just get it
        styles['InvoiceTitle'] = styles['Title']
        styles['InvoiceTitle'].fontSize = 16
        styles['InvoiceTitle'].alignment = TA_CENTER
        styles['InvoiceTitle'].spaceAfter = 12
    
    try:
        # Custom Heading style
        styles.add(ParagraphStyle(
            name='InvoiceHeading',
            fontSize=14,
            spaceAfter=10
        ))
    except KeyError:
        # Style already exists, use it
        styles['InvoiceHeading'] = styles['Heading1']
        styles['InvoiceHeading'].fontSize = 14
        styles['InvoiceHeading'].spaceAfter = 10
    
    try:
        # Right-aligned text
        styles.add(ParagraphStyle(
            name='Normal_RIGHT',
            parent=styles['Normal'],
            alignment=TA_RIGHT
        ))
    except KeyError:
        # Create a new style based on Normal
        styles['Normal_RIGHT'] = ParagraphStyle(
            'Normal_RIGHT',
            parent=styles['Normal'],
            alignment=TA_RIGHT
        )
    
    # Content elements
    elements = []
    
    # Invoice title
    elements.append(Paragraph(f"INVOICE #{invoice_data['invoice_number']}", styles['Title']))
    elements.append(Spacer(1, 20))
    
    # Company information
    company_data = [
        [Paragraph(f"<b>FROM:</b>", styles['Normal']), 
         Paragraph(f"<b>TO:</b>", styles['Normal'])],
        [Paragraph(f"{invoice_data['vendor']['name']}<br/>"
                   f"{invoice_data['vendor']['address']}<br/>"
                   f"Tax ID: {invoice_data['vendor']['tax_id']}", styles['Normal']),
         Paragraph(f"{invoice_data['customer']['name']}<br/>"
                   f"{invoice_data['customer']['address']}<br/>"
                   f"Tax ID: {invoice_data['customer']['tax_id']}", styles['Normal'])]
    ]
    company_table = Table(company_data, colWidths=[doc.width/2.0]*2)
    company_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(company_table)
    elements.append(Spacer(1, 20))
    
    # Invoice details
    invoice_details_data = [
        [Paragraph("<b>Invoice Date:</b>", styles['Normal']), 
         Paragraph(invoice_data['issue_date'], styles['Normal'])],
        [Paragraph("<b>Due Date:</b>", styles['Normal']), 
         Paragraph(invoice_data['due_date'], styles['Normal'])],
        [Paragraph("<b>Payment Terms:</b>", styles['Normal']), 
         Paragraph(invoice_data['payment_terms'], styles['Normal'])],
    ]
    invoice_details_table = Table(invoice_details_data, colWidths=[100, 150])
    invoice_details_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(invoice_details_table)
    elements.append(Spacer(1, 20))
    
    # Line items
    elements.append(Paragraph("Invoice Items", styles['Heading1']))
    
    # Line items table header
    line_items_data = [
        [Paragraph("<b>Description</b>", styles['Normal']),
         Paragraph("<b>Quantity</b>", styles['Normal_RIGHT']),
         Paragraph("<b>Unit Price</b>", styles['Normal_RIGHT']),
         Paragraph("<b>Amount</b>", styles['Normal_RIGHT'])]
    ]
    
    # Add line items
    for item in invoice_data['line_items']:
        line_items_data.append([
            Paragraph(item['description'], styles['Normal']),
            Paragraph(str(item['quantity']), styles['Normal_RIGHT']),
            Paragraph(f"${item['unit_price']:.2f}", styles['Normal_RIGHT']),
            Paragraph(f"${item['amount']:.2f}", styles['Normal_RIGHT'])
        ])
    
    # Create line items table
    col_widths = [doc.width-240, 60, 80, 100]
    line_items_table = Table(line_items_data, colWidths=col_widths)
    line_items_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(line_items_table)
    elements.append(Spacer(1, 10))
    
    # Totals
    totals_data = [
        ["", Paragraph("<b>Subtotal:</b>", styles['Normal_RIGHT']), 
         Paragraph(f"${invoice_data['subtotal']:.2f}", styles['Normal_RIGHT'])],
        ["", Paragraph(f"<b>Tax ({invoice_data['tax_rate']*100:.1f}%):</b>", styles['Normal_RIGHT']), 
         Paragraph(f"${invoice_data['tax_amount']:.2f}", styles['Normal_RIGHT'])],
        ["", Paragraph("<b>TOTAL:</b>", styles['Normal_RIGHT']), 
         Paragraph(f"<b>${invoice_data['total_amount']:.2f}</b>", styles['Normal_RIGHT'])]
    ]
    
    totals_table = Table(totals_data, colWidths=[doc.width-280, 180, 100])
    totals_table.setStyle(TableStyle([
        ('LINEBELOW', (1, 1), (2, 1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(totals_table)
    
    # Payment instructions
    elements.append(Spacer(1, 30))
    elements.append(Paragraph("Payment Instructions:", styles['Normal']))
    elements.append(Paragraph("Please include the invoice number on your payment.", styles['Normal']))
    elements.append(Paragraph(f"Make checks payable to {invoice_data['vendor']['name']}", styles['Normal']))
    
    # Build the PDF
    doc.build(elements)

def main():
    """Generate sample invoices based on command arguments."""
    parser = argparse.ArgumentParser(description="Generate sample invoice PDFs for Document AI training/testing")
    parser.add_argument("-n", "--num-invoices", type=int, default=5, help="Number of invoices to generate")
    parser.add_argument("-o", "--output-dir", type=str, default="./sample_data/invoices/training", 
                        help="Directory to save generated invoices")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate the invoices
    print(f"Generating {args.num_invoices} sample invoices in {args.output_dir}...")
    for i in range(args.num_invoices):
        invoice_data = generate_invoice_data()
        output_path = os.path.join(args.output_dir, f"sample_invoice_{i+1}.pdf")
        create_invoice_pdf(invoice_data, output_path)
        print(f"Created: {output_path}")
    
    print(f"Successfully generated {args.num_invoices} sample invoices.")

if __name__ == "__main__":
    main()
