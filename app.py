import gradio as gr
from query_pdf import query_pdf
import os
import tempfile
import shutil

def process_pdf_query(query, system_prompt, pdf_file, progress=gr.Progress()):
    """
    Process a PDF query using the Gradio interface.
    
    Args:
        query (str): The question to ask about the PDF
        system_prompt (str): Context for how to answer the question
        pdf_file (str): Path to the uploaded PDF file
        progress (gr.Progress): Gradio progress tracker
    
    Returns:
        str: Formatted output with the query results
    """
    if not query:
        return "Please enter a query."
    
    if not pdf_file:
        return "Please upload a PDF file."
    
    try:
        # Create a temporary file to store the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            # Copy the uploaded file to the temporary file
            shutil.copy2(pdf_file.name, tmp_file.name)
            tmp_path = tmp_file.name
        
        # Combine system prompt and query
        full_query = f"{system_prompt}\n\nQuery: {query}" if system_prompt else query
        
        # Process the query with progress updates
        result = query_pdf(
            full_query, 
            pdf_path=tmp_path,
            progress_callback=lambda p, msg: progress(p, desc=msg)
        )
        
        # Format the output
        output = f"System Context: {system_prompt}\n" if system_prompt else ""
        output += f"Query: {query}\n"
        output += f"Most relevant information found on page {result['page']}\n\n"
        
        output += "Summary:\n"
        output += result['summary'] + "\n\n"
        
        output += "Relevant content:\n"
        for i, sentence in enumerate(result['relevant_content'], 1):
            output += f"{i}. {sentence}\n"
        
        output += "\nContext from the page:\n"
        output += result['text'][:500] + "...\n\n"
        
        output += "Timing Information:\n"
        output += f"Total processing time: {result['timing']['total_time']:.2f} seconds\n"
        output += f"Average time per page: {result['timing']['avg_page_time']:.2f} seconds"
        
        # Clean up the temporary file
        os.unlink(tmp_path)
        
        return output
        
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

# Create the Gradio interface
with gr.Blocks(title="PDF Query Tool") as app:
    gr.Markdown("# PDF Query Tool")
    gr.Markdown("Upload a PDF and ask questions about its content.")
    
    with gr.Row():
        with gr.Column():
            # Input components
            system_prompt = gr.Textbox(
                label="System Context",
                placeholder="Provide context for how to answer the question (e.g., 'Focus on financial metrics' or 'Look for environmental impact data')",
                lines=3
            )
            query_input = gr.Textbox(
                label="Enter your query",
                placeholder="What would you like to know about the document?",
                lines=3
            )
            pdf_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"]
            )
            submit_btn = gr.Button("Process Query")
        
        with gr.Column():
            # Output component
            output = gr.Textbox(
                label="Results",
                lines=20,
                max_lines=20
            )
    
    # Set up the event handler
    submit_btn.click(
        fn=process_pdf_query,
        inputs=[system_prompt, query_input, pdf_input],
        outputs=output
    )

# Launch the app
if __name__ == "__main__":
    app.launch(share=True) 