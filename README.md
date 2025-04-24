# PDF Query Tool

A powerful tool that uses the Colpali engine to analyze PDF documents and answer questions about their content. The tool provides a user-friendly web interface for querying PDFs and getting detailed responses.

## Features

- Upload and analyze PDF documents
- Ask questions about the document content
- Get detailed responses with relevant page numbers and context
- System prompt to guide the model's response focus
- Progress tracking during document processing
- Timing information for performance monitoring

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:7860)

3. Using the interface:
   - **System Context**: (Optional) Enter instructions to guide the model's response focus
     - Example: "Focus on financial metrics" or "Look for environmental impact data"
   - **Query**: Enter your question about the PDF content
   - **Upload PDF**: Click to upload your PDF document
   - Click "Process Query" to analyze the document

4. The results will show:
   - The page number where relevant information was found
   - A summary of the findings
   - Relevant content from the document
   - Context from the page
   - Processing timing information

## Example Queries

- "What are the main financial metrics mentioned in the report?"
- "What are the environmental impact numbers?"
- "What are the key dates mentioned in the document?"
- "What are the main challenges discussed?"

## System Context Examples

- "Focus on numerical data and statistics"
- "Look for information about future plans and projections"
- "Pay attention to dates and timelines"
- "Focus on environmental impact data"

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for better performance)
- See requirements.txt for detailed package dependencies

## Notes

- The first run will download the Colpali model, which may take some time
- Processing time depends on the PDF size and complexity
- The tool works best with well-structured PDFs containing text (not scanned images)

## Troubleshooting

If you encounter any issues:

1. Check your internet connection (required for model download)
2. Ensure you have sufficient disk space for the model
3. Verify that your PDF is not corrupted and contains extractable text
4. Check the console output for any error messages

## License

[Add your license information here] 