import gradio as gr
import fitz  # PyMuPDF
import re
from datetime import datetime
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import tempfile
import shutil
from colpali_engine.models import ColPali, ColPaliProcessor
import torch
import json

# XBRL namespace definitions
XBRL_NAMESPACES = {
    'xbrli': 'http://www.xbrl.org/2003/instance',
    'xbrldi': 'http://xbrl.org/2006/xbrldi',
    'xbrldt': 'http://xbrl.org/2005/xbrldt',
    'iso4217': 'http://www.xbrl.org/2003/iso4217',
    'xlink': 'http://www.w3.org/1999/xlink',
    'link': 'http://www.xbrl.org/2003/linkbase',
    'xsi': 'http://www.w3.org/2001/XMLSchema-instance',
    'esg': 'http://www.example.com/xbrl/esg'
}

# Define metric units and their XBRL representations
METRIC_UNITS = {
    'emissions': {
        'unit': 'kt',
        'description': 'kilotons of CO₂e',
        'namespace': 'esg'
    },
    'energy_use': {
        'unit': 'GWh',
        'description': 'Gigawatt hours',
        'namespace': 'esg'
    },
    'water_use': {
        'unit': 'ML',
        'description': 'Megaliters',
        'namespace': 'esg'
    },
    'waste': {
        'unit': 'kt',
        'description': 'kilotons',
        'namespace': 'esg'
    }
}

class XBRLConverter:
    def __init__(self):
        # Initialize Colpali model for text extraction
        self.model = ColPali.from_pretrained(
            "vidore/colpali-v1.2-merged",
            torch_dtype=torch.bfloat16,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(
            "vidore/colpali-v1.2-merged",
            trust_remote_code=True
        )
        
        # Load XBRL taxonomy mappings
        self.taxonomy_mappings = self._load_taxonomy_mappings()
    
    def _load_taxonomy_mappings(self):
        """Load XBRL taxonomy mappings from a JSON file."""
        return {
            'scope1_emissions': {
                'keywords': ['scope 1', 'direct emissions', 'direct greenhouse gas', 'direct GHG'],
                'unit': 'kt',
                'context': 'environmental'
            },
            'scope2_emissions': {
                'keywords': ['scope 2', 'indirect emissions', 'indirect greenhouse gas', 'indirect GHG'],
                'unit': 'kt',
                'context': 'environmental'
            },
            'total_ghg_emissions': {
                'keywords': ['total emissions', 'total greenhouse gas', 'total GHG', 'total carbon'],
                'unit': 'kt',
                'context': 'environmental'
            },
            'energy_from_electricity': {
                'keywords': ['electricity consumption', 'electricity usage', 'electricity energy'],
                'unit': 'GWh',
                'context': 'environmental'
            },
            'energy_from_fossil_fuels': {
                'keywords': ['fossil fuel consumption', 'fossil fuel usage', 'coal consumption'],
                'unit': 'GWh',
                'context': 'environmental'
            },
            'freshwater_withdrawal': {
                'keywords': ['freshwater withdrawal', 'water withdrawal', 'water intake'],
                'unit': 'ML',
                'context': 'environmental'
            },
            'water_efficiency': {
                'keywords': ['water efficiency', 'water reuse', 'water recycling'],
                'unit': 'Pure',
                'context': 'environmental'
            },
            'waste_generated': {
                'keywords': ['waste generated', 'total waste', 'waste production'],
                'unit': 'kt',
                'context': 'environmental'
            },
            'waste_recycled': {
                'keywords': ['waste recycled', 'recycling rate', 'waste recovery'],
                'unit': 'kt',
                'context': 'environmental'
            }
        }
    
    def extract_structured_data(self, text):
        """Extract structured data from text using Colpali."""
        data = {}
        years = set()
        
        # Extract years mentioned in the text
        year_pattern = r'\b(20\d{2})\b'
        years.update(re.findall(year_pattern, text))
        
        # Extract numbers and their context
        number_pattern = r'([$]?\d+(?:,\d{3})*(?:\.\d+)?)'
        sentences = re.split(r'[.!?]+', text)
        
        # Track processed metrics to avoid duplicates
        processed_metrics = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Try to identify the year from the sentence
            year_match = re.search(year_pattern, sentence)
            year = year_match.group(1) if year_match else None
            
            if not year:
                continue
            
            # Look for numbers in the sentence
            numbers = re.findall(number_pattern, sentence)
            if not numbers:
                continue
            
            # Try to categorize the sentence based on keywords
            for metric, info in self.taxonomy_mappings.items():
                if any(keyword in sentence.lower() for keyword in info['keywords']):
                    # Skip if we've already processed this metric for this year
                    if (metric, year) in processed_metrics:
                        continue
                    
                    if metric not in data:
                        data[metric] = {}
                    if year not in data[metric]:
                        data[metric][year] = []
                    
                    # Extract the most relevant number (usually the first one)
                    value = numbers[0].replace(',', '')
                    try:
                        value = float(value)
                        
                        # Validate the value based on the metric type
                        if metric in ['scope1_emissions', 'scope2_emissions', 'total_ghg_emissions']:
                            if 0 <= value <= 1000000:  # Reasonable range for emissions in kt
                                data[metric][year].append({
                                    'value': value,
                                    'unit': info['unit'],
                                    'context': info['context']
                                })
                                processed_metrics.add((metric, year))
                        elif metric in ['energy_from_electricity', 'energy_from_fossil_fuels']:
                            if 0 <= value <= 100000:  # Reasonable range for energy in GWh
                                data[metric][year].append({
                                    'value': value,
                                    'unit': info['unit'],
                                    'context': info['context']
                                })
                                processed_metrics.add((metric, year))
                        elif metric in ['freshwater_withdrawal', 'waste_generated', 'waste_recycled']:
                            if 0 <= value <= 1000000:  # Reasonable range for water/waste in ML/kt
                                data[metric][year].append({
                                    'value': value,
                                    'unit': info['unit'],
                                    'context': info['context']
                                })
                                processed_metrics.add((metric, year))
                        elif metric == 'water_efficiency':
                            if 0 <= value <= 100:  # Percentage range
                                data[metric][year].append({
                                    'value': value,
                                    'unit': info['unit'],
                                    'context': info['context']
                                })
                                processed_metrics.add((metric, year))
                    except ValueError:
                        continue
        
        return data
    
    def create_xbrl_document(self, data, company_name, report_date):
        """Create an XBRL document from the extracted data."""
        # Register all namespaces first
        for prefix, uri in XBRL_NAMESPACES.items():
            ET.register_namespace(prefix, uri)
        
        # Create the root element
        root = ET.Element('xbrl')
        
        # Add contexts for each year
        contexts = {}
        for metric_data in data.values():
            for year in metric_data.keys():
                if year not in contexts:
                    context = ET.SubElement(root, '{' + XBRL_NAMESPACES['xbrli'] + '}context', {'id': f'D-{year}'})
                    period = ET.SubElement(context, '{' + XBRL_NAMESPACES['xbrli'] + '}period')
                    start_date = ET.SubElement(period, '{' + XBRL_NAMESPACES['xbrli'] + '}startDate')
                    start_date.text = f"{year}-01-01"
                    end_date = ET.SubElement(period, '{' + XBRL_NAMESPACES['xbrli'] + '}endDate')
                    end_date.text = f"{year}-12-31"
                    
                    company = ET.SubElement(context, '{' + XBRL_NAMESPACES['xbrli'] + '}entity')
                    company_id = ET.SubElement(company, '{' + XBRL_NAMESPACES['xbrli'] + '}identifier', {'scheme': 'http://www.example.com/company'})
                    company_id.text = company_name
                    contexts[year] = context
        
        # Add units for each metric type
        units = {}
        for metric, info in self.taxonomy_mappings.items():
            unit_id = f'U-{info["unit"]}'
            if unit_id not in units:
                unit = ET.SubElement(root, '{' + XBRL_NAMESPACES['xbrli'] + '}unit', {'id': unit_id})
                measure = ET.SubElement(unit, '{' + XBRL_NAMESPACES['xbrli'] + '}measure')
                measure.text = info['unit']
                units[unit_id] = unit
        
        # Add labels for metrics
        labels = {
            'scope1_emissions': 'Scope 1 Greenhouse Gas Emissions',
            'scope2_emissions': 'Scope 2 Greenhouse Gas Emissions',
            'total_ghg_emissions': 'Total Greenhouse Gas Emissions',
            'energy_from_electricity': 'Energy Consumption from Electricity',
            'energy_from_fossil_fuels': 'Energy Consumption from Fossil Fuels',
            'freshwater_withdrawal': 'Freshwater Withdrawal',
            'water_efficiency': 'Water Efficiency',
            'waste_generated': 'Total Waste Generated',
            'waste_recycled': 'Waste Recycled'
        }
        
        # Add facts
        for metric, year_data in data.items():
            for year, values in year_data.items():
                for value_info in values:
                    fact = ET.SubElement(root, '{' + XBRL_NAMESPACES['esg'] + '}' + metric)
                    fact.text = str(value_info['value'])
                    fact.set('contextRef', f'D-{year}')
                    fact.set('unitRef', f'U-{value_info["unit"]}')
                    fact.set('decimals', '2')  # Add precision information
                    
                    # Add label
                    if metric in labels:
                        label = ET.SubElement(root, '{' + XBRL_NAMESPACES['link'] + '}label', {
                            '{' + XBRL_NAMESPACES['xlink'] + '}type': 'resource',
                            '{' + XBRL_NAMESPACES['xlink'] + '}label': f'label_{metric}',
                            '{' + XBRL_NAMESPACES['xlink'] + '}role': 'http://www.xbrl.org/2003/role/label'
                        })
                        label.text = labels[metric]
        
        # Convert to pretty XML with proper namespace handling
        xml_str = ET.tostring(root, encoding='unicode', method='xml')
        dom = minidom.parseString(xml_str)
        return dom.toprettyxml(indent="  ")

def convert_to_xbrl(pdf_file, company_name, report_date, progress=gr.Progress()):
    """Convert PDF to XBRL format."""
    try:
        # Create temporary file for PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            shutil.copy2(pdf_file.name, tmp_file.name)
            tmp_path = tmp_file.name
        
        # Initialize converter
        converter = XBRLConverter()
        
        # Extract text from PDF
        doc = fitz.open(tmp_path)
        text = ""
        total_pages = len(doc)
        
        for i, page in enumerate(doc):
            progress(i / total_pages, f"Processing page {i+1} of {total_pages}")
            text += page.get_text()
        
        # Extract structured data
        progress(0.8, "Extracting structured data...")
        data = converter.extract_structured_data(text)
        
        # Create XBRL document
        progress(0.9, "Creating XBRL document...")
        xbrl_doc = converter.create_xbrl_document(data, company_name, report_date)
        
        # Clean up
        os.unlink(tmp_path)
        
        return xbrl_doc
        
    except Exception as e:
        return f"Error converting PDF to XBRL: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="PDF to XBRL Converter") as app:
    gr.Markdown("# PDF to XBRL Converter")
    gr.Markdown("Convert PDF documents to XBRL format for financial reporting.")
    
    with gr.Row():
        with gr.Column():
            # Input components
            pdf_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"]
            )
            company_name = gr.Textbox(
                label="Company Name",
                placeholder="Enter company name"
            )
            report_date = gr.Textbox(
                label="Report Date",
                placeholder="YYYY-MM-DD",
                value=datetime.now().strftime("%Y-%m-%d")
            )
            convert_btn = gr.Button("Convert to XBRL")
        
        with gr.Column():
            # Output component
            output = gr.Textbox(
                label="XBRL Output",
                lines=20,
                max_lines=20
            )
    
    # Set up the event handler
    convert_btn.click(
        fn=convert_to_xbrl,
        inputs=[pdf_input, company_name, report_date],
        outputs=output
    )

# Launch the app
if __name__ == "__main__":
    app.launch(share=True) 