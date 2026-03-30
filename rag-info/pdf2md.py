import warnings
warnings.filterwarnings("ignore")

from transformers import NougatProcessor, VisionEncoderDecoderModel, BertTokenizer, BertForNextSentencePrediction
import fitz  # PyMuPDF
from PIL import Image
import torch
from pathlib import Path
import json
import time
from tqdm import tqdm

# Configuration
PDF_DIR = "./pdf"  # Directory containing PDF files
MD_OUTPUT_DIR = "./output"  # Output directory for Markdown files
PROCESSED_LOG = "./output/processed_files.json"  # Log of processed files

# Initialize Nougat model (load once)
processor = NougatProcessor.from_pretrained("facebook/nougat-base")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Initialize BERT for continuity check (load once)
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
bert_model.to(device)

print("BERT model loaded for continuity checking")
print(f"Using device: {device}")

def load_processed_files():
    """Load set of already processed filenames"""
    try:
        with open(PROCESSED_LOG, "r") as f:
            return set(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()

def save_processed_file(filename):
    """Add filename to processed files log"""
    processed = load_processed_files()
    processed.add(filename)
    with open(PROCESSED_LOG, "w") as f:
        json.dump(list(processed), f)

def get_single_page_image(pdf_path, page_num, dpi=300):
    """Extract single page as PIL image without loading full PDF"""
    with fitz.open(pdf_path) as doc:
        if page_num < 0 or page_num >= len(doc):
            raise ValueError(f"Page number {page_num} out of range")
        pix = doc[page_num].get_pixmap(dpi=dpi)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def process_page(image):
    """Process single image to Markdown with timing"""
    start_time = time.time()
    
    # Preprocessing
    pixel_values = processor(image, return_tensors="pt").pixel_values
    
    # Generation
    outputs = model.generate(
        pixel_values.to(device),
        min_length=1,
        max_new_tokens=2000,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
    )
    
    # Post-processing
    sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    markdown = processor.post_process_generation(sequence, fix_markdown=True)
    
    elapsed = time.time() - start_time
    return markdown, elapsed

def are_continuous(md_i, md_j, M=100):
    """
    Check if two Markdown strings are continuous using BERT's next sentence prediction.
    Args:
        md_i (str): Markdown content of the previous page
        md_j (str): Markdown content of the next page
        M (int): Number of tokens to take from each end
    Returns:
        bool: True if continuous, False otherwise
    """
    # Early return for empty strings
    if not md_i.strip() or not md_j.strip():
        return False
    
    # Tokenize the Markdown strings
    tokens_i = bert_tokenizer.tokenize(md_i)
    tokens_j = bert_tokenizer.tokenize(md_j)
    
    # Take the last M tokens of md_i and the first M tokens of md_j
    tail_i = tokens_i[-M:] if len(tokens_i) > M else tokens_i
    head_j = tokens_j[:M] if len(tokens_j) > M else tokens_j
    
    # Create input for BERT
    input_ids = bert_tokenizer.encode(
        tail_i,
        head_j,
        add_special_tokens=True,
        max_length=512,
        truncation=True
    )
    
    # Convert to tensor and move to device
    input_ids = torch.tensor([input_ids]).to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = bert_model(input_ids)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        is_next_prob = probs[0, 0].item()  # Probability of being next
    
    return is_next_prob > 0.5

def process_pdf(pdf_path, specific_page=None):
    """Process PDF file to Markdown with progress tracking and continuity checking"""
    try:
        pdf_name = Path(pdf_path).stem
        print(f"\nProcessing: {pdf_name}")
        
        # Open PDF just to get page count
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            
            # Determine pages to process
            if specific_page is not None:
                if not (0 <= specific_page < total_pages):
                    print(f"Warning: Page {specific_page} out of range, processing all")
                    page_range = range(total_pages)
                else:
                    page_range = [specific_page]
            else:
                page_range = range(total_pages)
        
        page_markdowns = []
        pdf_to_image_time = 0
        image_to_md_time = 0
        
        with tqdm(page_range, desc="Pages", unit="page") as pbar:
            for page_num in pbar:
                # PDF to Image conversion
                start = time.time()
                try:
                    image = get_single_page_image(pdf_path, page_num)
                    pdf_to_image_time += time.time() - start
                except Exception as e:
                    print(f"\nError processing page {page_num}: {str(e)}")
                    continue
                
                # Image to Markdown conversion
                md_content, elapsed = process_page(image)
                image_to_md_time += elapsed
                
                # Store Markdown without trailing newlines
                page_markdowns.append(md_content.rstrip())
                
                # Update progress bar with timing info
                pbar.set_postfix({
                    'pdf2img': f'{pdf_to_image_time/(pbar.n+1):.1f}s/page',
                    'img2md': f'{image_to_md_time/(pbar.n+1):.1f}s/page'
                })
        
        # Check continuity and join pages 
        if len(page_markdowns) == 0:
            print("No pages processed")
            return False
        elif len(page_markdowns) == 1:
            full_markdown = page_markdowns[0]
        else:
            full_markdown = page_markdowns[0]
            for i in range(1, len(page_markdowns)):
                if are_continuous(page_markdowns[i-1], page_markdowns[i]):
                    full_markdown += " " + page_markdowns[i]  # Continuous 
                else:
                    full_markdown += "\n\n" + page_markdowns[i]  # Non-Continuous
        
        # Save results
        Path(MD_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        md_path = Path(MD_OUTPUT_DIR) / f"{pdf_name}.md"
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(full_markdown)
        
        save_processed_file(pdf_name)
        
        print(f"\nCompleted: {md_path}")
        print(f"Timing summary:")
        print(f"- PDF to Image: {pdf_to_image_time:.2f}s total ({pdf_to_image_time/len(page_range):.2f}s/page)")
        print(f"- Image to Markdown: {image_to_md_time:.2f}s total ({image_to_md_time/len(page_range):.2f}s/page)")
        return True
    
    except Exception as e:
        print(f"\nError processing {pdf_name}: {str(e)}")
        return False

def main():
    # Prepare directories
    Path(PDF_DIR).mkdir(parents=True, exist_ok=True)
    Path(MD_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Get unprocessed PDFs
    processed_files = load_processed_files()
    pdf_files = [f for f in Path(PDF_DIR).glob("*.pdf") 
                if f.stem not in processed_files]
    
    print(f"Found {len(pdf_files)} unprocessed PDF files")
    
    if pdf_files:
        # Process all pdf files
        for pdf_path in pdf_files:
            try:
                process_pdf(pdf_path)  # Process all pages
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")

if __name__ == "__main__":
    main()