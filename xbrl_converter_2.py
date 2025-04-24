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
        'unit': 'GJ',  # Changed from GWh to GJ to match report
        'description': 'Gigajoules',
        'namespace': 'esg'
    },
    'water_use': {
        'unit': 'ML',
        'description': 'Megaliters',
        'namespace': 'esg'
    },
    'waste': {
        'unit': 'tonnes',  # Changed from kt to tonnes to match report
        'description': 'tonnes',
        'namespace': 'esg'
    },
    'percentage': {
        'unit': 'Pure',
        'description': 'Percentage',
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
                'patterns': [r'Scope 1\s+(\d+)', r'direct emissions.*?(\d+)\s*kt'],
                'table_headers': ['Scope 1', 'Direct emissions'],
                'unit': 'kt',
                'context': 'environmental'
            },
            'scope2_emissions': {
                'keywords': ['scope 2', 'indirect emissions', 'indirect greenhouse gas', 'indirect GHG'],
                'patterns': [r'Scope 2\s+(\d+)', r'indirect emissions.*?(\d+)\s*kt'],
                'table_headers': ['Scope 2', 'Indirect emissions'],
                'unit': 'kt',
                'context': 'environmental'
            },
            'total_ghg_emissions': {
                'keywords': ['total emissions', 'total greenhouse gas', 'total GHG', 'total carbon', 'total scope 1 and 2'],
                'patterns': [r'Total scope 1 and 2 emissions\s+(\d+)', r'Total GHG emissions.*?(\d+)\s*kt'],
                'table_headers': ['Total GHG emissions', 'Total scope 1 and 2'],
                'unit': 'kt',
                'context': 'environmental'
            },
            'energy_from_electricity': {
                'keywords': ['electricity consumption', 'electricity usage', 'energy from electricity'],
                'patterns': [r'Energy from electricity.*?(\d+(?:\.\d+)?)\s*million GJ', r'Electricity consumption.*?(\d+(?:\.\d+)?)\s*MWh'],
                'table_headers': ['Energy from electricity', 'Electricity consumption'],
                'unit': 'GJ',
                'context': 'environmental'
            },
            'total_energy_used': {
                'keywords': ['total energy', 'energy used', 'energy consumption'],
                'patterns': [r'Total energy used.*?(\d+(?:\.\d+)?)\s*million GJ', r'Total energy consumption.*?(\d+(?:\.\d+)?)\s*GJ'],
                'table_headers': ['Total energy used', 'Total energy'],
                'unit': 'GJ',
                'context': 'environmental'
            },
            'freshwater_withdrawal': {
                'keywords': ['freshwater withdrawal', 'freshwater abstraction', 'water withdrawal'],
                'patterns': [r'Freshwater abstraction.*?(\d+)\s*ML', r'Freshwater withdrawal.*?(\d+)\s*megalitres'],
                'table_headers': ['Freshwater withdrawal', 'Freshwater abstraction'],
                'unit': 'ML',
                'context': 'environmental'
            },
            'water_efficiency': {
                'keywords': ['water efficiency', 'water reuse', 'water recycling', 'water reused/recycled'],
                'patterns': [r'Water reused/recycled.*?(\d+)%', r'Water efficiency.*?(\d+)%'],
                'table_headers': ['Water efficiency', 'Water reused/recycled'],
                'unit': 'Pure',
                'context': 'environmental'
            },
            'hazardous_waste': {
                'keywords': ['hazardous waste', 'hazardous waste to legal landfill'],
                'patterns': [r'Hazardous waste to legal landfill.*?(\d+)\s*tonnes', r'Hazardous waste generated.*?(\d+)\s*tonnes'],
                'table_headers': ['Hazardous waste', 'Hazardous waste to legal landfill'],
                'unit': 'tonnes',
                'context': 'environmental'
            },
            'non_hazardous_waste': {
                'keywords': ['non-hazardous waste', 'non hazardous waste', 'non-hazardous waste to legal landfill'],
                'patterns': [r'Non-hazardous waste to legal landfill.*?(\d+)\s*tonnes', r'Non-hazardous waste generated.*?(\d+)\s*tonnes'],
                'table_headers': ['Non-hazardous waste', 'Non-hazardous waste to legal landfill'],
                'unit': 'tonnes',
                'context': 'environmental'
            },
            'fatalities': {
                'keywords': ['fatality', 'fatalities', 'work-related fatal injuries'],
                'patterns': [r'Fatality\s*(\d+)', r'work-related fatal injuries\s*(\d+)'],
                'table_headers': ['Fatality', 'Fatalities'],
                'unit': 'Pure',
                'context': 'social'
            },
            'lost_time_injury_frequency_rate': {
                'keywords': ['lost-time injury frequency rate', 'LTIFR', 'lost time injuries frequency'],
                'patterns': [r'LTIFR.*?(\d+\.\d+)', r'Lost-time injury frequency rate.*?(\d+\.\d+)'],
                'table_headers': ['LTIFR', 'Lost-time injury frequency rate'],
                'unit': 'Pure',
                'context': 'social'
            },
            'total_recordable_case_frequency_rate': {
                'keywords': ['TRCFR', 'total recordable case frequency rate'],
                'patterns': [r'TRCFR.*?(\d+\.\d+)', r'Total recordable case frequency rate.*?(\d+\.\d+)'],
                'table_headers': ['TRCFR', 'Total recordable case frequency rate'],
                'unit': 'Pure',
                'context': 'social'
            }
        }
    
    def extract_structured_data(self, text):
        """Extract structured data from text using Colpali and pattern matching."""
        data = {}
        
        # Extract tables from the text
        tables = self._extract_tables(text)
        
        # Process tables to find performance indicators
        table_data = self._process_tables(tables)
        
        # Extract patterns from text
        text_data = self._extract_patterns_from_text(text)
        
        # Merge table data and text data
        merged_data = self._merge_data(table_data, text_data)
        
        return merged_data
    
    def _extract_tables(self, text):
        """Extract tables from the text."""
        # Look for common table patterns in ESG reports
        # This is a simplified approach - a more robust implementation would use Colpali's 
        # document structure extraction capabilities
        tables = []
        
        # Look for sections that might contain tables
        performance_sections = re.findall(r'(Performance.*?)\n\n', text, re.DOTALL)
        for section in performance_sections:
            # Try to identify table rows
            rows = re.findall(r'(\w+(?:\s+\w+)*)[\s\|]+(\d{4})[\s\|]+(\d{4})[\s\|]+(\d{4})[\s\|]+(\d{4})', section)
            if rows:
                tables.append({
                    'header': ['Metric', '2023', '2022', '2021', '2020'],
                    'rows': rows
                })
        
        return tables
    
    def _process_tables(self, tables):
        """Process tables to extract performance indicators."""
        data = {}
        
        for table in tables:
            for row in table['rows']:
                metric = row[0].strip().lower()
                
                # Try to map the table row to a known metric
                mapped_metric = None
                for metric_key, info in self.taxonomy_mappings.items():
                    if any(header.lower() in metric for header in info['table_headers']):
                        mapped_metric = metric_key
                        break
                
                if not mapped_metric:
                    continue
                
                # Extract values for each year
                for i, year in enumerate(['2023', '2022', '2021', '2020']):
                    if i + 1 < len(row):
                        try:
                            value = float(row[i+1].replace(',', ''))
                            
                            if mapped_metric not in data:
                                data[mapped_metric] = {}
                            if year not in data[mapped_metric]:
                                data[mapped_metric][year] = []
                                
                            data[mapped_metric][year].append({
                                'value': value,
                                'unit': self.taxonomy_mappings[mapped_metric]['unit'],
                                'context': self.taxonomy_mappings[mapped_metric]['context']
                            })
                        except ValueError:
                            continue
        
        return data
    
    def _extract_patterns_from_text(self, text):
        """Extract patterns from text."""
        data = {}
        
        # Extract years from text
        years = self._extract_years(text)
        
        # Filter out years that are likely not relevant
        current_year = datetime.now().year
        valid_years = [year for year in years if 2000 <= int(year) <= current_year + 1]
        
        # For each metric, try to find matching patterns
        for metric, info in self.taxonomy_mappings.items():
            if 'patterns' not in info:
                continue
                
            for pattern in info['patterns']:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    value = float(match.replace(',', ''))
                    
                    # Try to find the year associated with this value
                    sentence = self._find_sentence_containing(text, match)
                    if not sentence:
                        continue
                        
                    year = self._extract_year_from_sentence(sentence, valid_years)
                    if not year:
                        continue
                    
                    if metric not in data:
                        data[metric] = {}
                    if year not in data[metric]:
                        data[metric][year] = []
                        
                    data[metric][year].append({
                        'value': value,
                        'unit': info['unit'],
                        'context': info['context']
                    })
        
        return data
    
    def _extract_years(self, text):
        """Extract all years mentioned in the text."""
        year_pattern = r'\b(20\d{2})\b'
        return set(re.findall(year_pattern, text))
    
    def _find_sentence_containing(self, text, substring):
        """Find the sentence containing a specific substring."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            if substring in sentence:
                return sentence
        return None
    
    def _extract_year_from_sentence(self, sentence, valid_years):
        """Extract a year from a sentence, ensuring it's in the valid years list."""
        year_pattern = r'\b(20\d{2})\b'
        years = re.findall(year_pattern, sentence)
        
        for year in years:
            if year in valid_years:
                return year
        
        # If no valid year was found, return the most recent valid year
        if valid_years:
            return max(valid_years)
        
        return None
    
    def _merge_data(self, table_data, text_data):
        """Merge data from tables and text patterns."""
        merged_data = {}
        
        # First, add all table data
        for metric, year_data in table_data.items():
            merged_data[metric] = year_data
        
        # Then add text data, but only if it doesn't conflict with table data
        for metric, year_data in text_data.items():
            if metric not in merged_data:
                merged_data[metric] = year_data
            else:
                for year, values in year_data.items():
                    if year not in merged_data[metric]:
                        merged_data[metric][year] = values
        
        return merged_data
    
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
            'total_energy_used': 'Total Energy Consumption',
            'freshwater_withdrawal': 'Freshwater Withdrawal',
            'water_efficiency': 'Water Efficiency Percentage',
            'hazardous_waste': 'Hazardous Waste to Legal Landfill',
            'non_hazardous_waste': 'Non-Hazardous Waste to Legal Landfill',
            'fatalities': 'Work-Related Fatalities',
            'lost_time_injury_frequency_rate': 'Lost Time Injury Frequency Rate',
            'total_recordable_case_frequency_rate': 'Total Recordable Case Frequency Rate'
        }
        
        # Add facts - only use the first value for each metric/year to avoid duplicates
        for metric, year_data in data.items():
            for year, values in year_data.items():
                if values:
                    value_info = values[0]  # Use the first value only
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

    def process_pdf_with_colpali(self, pdf_path):
        """Process PDF using Colpali to extract structured data."""
        # This is a placeholder for integrating with Colpali
        # In a real implementation, you would use Colpali to extract
        # structured data from the PDF, including tables
        try:
            # For demonstration, we'll just extract text using PyMuPDF
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            print(f"Error processing PDF with Colpali: {str(e)}")
            return ""

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
    gr.Markdown("Convert PDF ESG reports to XBRL format for standardized reporting.")
    
    with gr.Row():
        with gr.Column():
            # Input components
            pdf_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"]
            )
            company_name = gr.Textbox(
                label="Company Name",
                placeholder="Enter company name",
                value="Thungela"  # Default to Thungela
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