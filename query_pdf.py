import os
import torch
from colpali_engine.models import ColPali, ColPaliProcessor
import requests
from requests.exceptions import RequestException
from PIL import Image
import fitz  # PyMuPDF for PDF processing
import gc
import re
import time
from datetime import timedelta

def extract_relevant_content(text, query):
    """Extract relevant content from text based on the query."""
    # Convert query to lowercase for case-insensitive matching
    query_terms = query.lower().split()
    
    # Find sentences that contain query terms
    sentences = re.split(r'[.!?]+', text)
    relevant_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if any(term in sentence.lower() for term in query_terms):
            relevant_sentences.append(sentence)
    
    return relevant_sentences

def generate_summary(query, page, text, relevant_content):
    """Generate a summary paragraph of the findings."""
    if not relevant_content:
        return f"No specific information related to your query was found on page {page}."
    
    # Generate summary
    summary = f"Based on the analysis of page {page}, here's what was found regarding your query:\n\n"
    
    # Add relevant content
    for i, sentence in enumerate(relevant_content, 1):
        summary += f"{i}. {sentence}\n"
    
    return summary

def query_pdf(query: str, pdf_path: str = "data/SASOL Climate Change Report 2023.pdf", batch_size: int = 1, progress_callback=None):
    """
    Query a PDF file using the Colpali engine.
    
    Args:
        query (str): The question or query to ask about the PDF content
        pdf_path (str): Path to the PDF file (defaults to the SASOL report)
        batch_size (int): Number of pages to process at once (default: 1 to minimize memory usage)
        progress_callback (callable): Optional callback function to report progress
    
    Returns:
        dict: Dictionary containing the page number, extracted content, and timing information
    """
    start_time = time.time()
    page_times = []
    
    try:
        # Initialize Colpali
        model_name = "vidore/colpali-v1.2-merged"
        model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()
        processor = ColPaliProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
    except RequestException as e:
        raise ConnectionError(
            "Failed to connect to Hugging Face to download the model. "
            "Please check your internet connection and try again. "
            f"Error details: {str(e)}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Colpali: {str(e)}")
    
    # Load the PDF
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    
    try:
        # Convert PDF pages to images and process in batches
        doc = fitz.open(pdf_path)
        page_scores = []
        total_pages = len(doc)
        
        # Process query once
        batch_queries = processor.process_queries([query]).to(model.device)
        with torch.no_grad():
            query_embeddings = model(**batch_queries)
        
        # Process pages in batches
        for i in range(0, total_pages, batch_size):
            batch_start_time = time.time()
            
            # Report progress
            if progress_callback:
                progress_callback(i / total_pages, f"Processing page {i+1} of {total_pages}")
            
            batch_images = []
            for j in range(i, min(i + batch_size, total_pages)):
                page = doc[j]
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                batch_images.append(img)
            
            # Process batch
            batch_images_processed = processor.process_images(batch_images).to(model.device)
            with torch.no_grad():
                image_embeddings = model(**batch_images_processed)
                scores = processor.score_multi_vector(query_embeddings, image_embeddings)
                page_scores.extend(scores.cpu().numpy())
            
            # Clear memory
            del batch_images_processed, image_embeddings, scores
            torch.cuda.empty_cache()
            gc.collect()
            
            # Record batch processing time
            batch_time = time.time() - batch_start_time
            page_times.append(batch_time)
        
        # Find the most relevant page
        most_relevant_page = max(range(len(page_scores)), key=lambda i: page_scores[i])
        
        # Extract text from the most relevant page
        page = doc[most_relevant_page]
        text = page.get_text()
        
        # Extract relevant content based on the query
        relevant_content = extract_relevant_content(text, query)
        
        # Generate summary
        summary = generate_summary(query, most_relevant_page + 1, text, relevant_content)
        
        total_time = time.time() - start_time
        
        # Report completion
        if progress_callback:
            progress_callback(1.0, "Analysis complete")
        
        return {
            'page': most_relevant_page + 1,
            'text': text,
            'relevant_content': relevant_content,
            'summary': summary,
            'timing': {
                'total_time': total_time,
                'page_times': page_times,
                'avg_page_time': sum(page_times) / len(page_times) if page_times else 0
            }
        }
        
    except Exception as e:
        raise RuntimeError(f"Error processing PDF: {str(e)}")
    finally:
        # Clean up
        if 'model' in locals():
            del model
        if 'processor' in locals():
            del processor
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    # Example usage
    query = "What are the numbers for the scope 1-2 emission vis. the scope 3 emissions?"
    try:
        result = query_pdf(query)
        print(f"Query: {query}")
        print(f"Most relevant information found on page {result['page']}")
        
        # Print summary
        print("\nSummary:")
        print(result['summary'])
        
        print("\nRelevant content:")
        for i, sentence in enumerate(result['relevant_content'], 1):
            print(f"{i}. {sentence}")
        
        print("\nContext from the page:")
        print(result['text'][:500] + "...")  # Print first 500 characters of context
        
        # Print timing information
        print("\nTiming Information:")
        print(f"Total processing time: {timedelta(seconds=int(result['timing']['total_time']))}")
        print(f"Average time per page: {timedelta(seconds=int(result['timing']['avg_page_time']))}")
        # print("\nTime per page:")
        # for i, page_time in enumerate(result['timing']['page_times']):
        #     print(f"Page {i+1}: {timedelta(seconds=int(page_time))}")
            
    except ConnectionError as e:
        print(f"Connection Error: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Check your internet connection")
        print("2. Verify you can access huggingface.co in your browser")
        print("3. If using a proxy, ensure it's properly configured")
    except Exception as e:
        print(f"Error: {str(e)}") 