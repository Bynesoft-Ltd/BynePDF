# BynePDF API Documentation

## PDFQueryProcessor

The main class for processing PDFs, embedding their content, and optionally answering queries.

### `__init__(minicpm_path: str = 'openbmb/MiniCPM-V-2_6', device: Optional[str] = None, verbose: bool = False, load_minicpm: bool = True)`

Initialize the PDFQueryProcessor.

- `minicpm_path`: Path to the MiniCPM model. Defaults to 'openbmb/MiniCPM-V-2_6'.
- `device`: Device to use for computation ('cuda' or 'cpu'). If None, will use CUDA if available.
- `verbose`: Whether to print verbose output.
- `load_minicpm`: Whether to load the MiniCPM model for question answering. If False, the pipeline will return raw text instead of answers.

Note: The LayoutLM-Byne model is always used and cannot be substituted.

### `_get_device(device: Optional[str]) -> torch.device`

Internal method to determine the appropriate device for computation.

- `device`: Optional string specifying the device ('cuda' or 'cpu').
- Returns: A `torch.device` object.

### `split_pdf_to_images(pdf_path: str) -> List[Image.Image]`

Convert a PDF file to a list of images.

- `pdf_path`: Path to the PDF file.
- Returns: List of PIL Image objects, one for each page of the PDF.

### `normalize_box(box: List[int], width: int, height: int) -> List[int]`

Static method to normalize bounding box coordinates.

- `box`: List of [x0, y0, x1, y1] coordinates.
- `width`: Width of the image.
- `height`: Height of the image.
- Returns: Normalized box coordinates.

### `apply_tesseract(image: Image.Image) -> Tuple[List[str], List[List[int]]]`

Apply OCR to an image and return words and bounding boxes.

- `image`: PIL Image object to process.
- Returns: A tuple containing a list of words and a list of normalized bounding boxes.

### `preprocess_for_model(words: List[str], boxes: List[List[int]], max_seq_length: int = 512, pad_token_box: List[int] = [0, 0, 0, 0]) -> Dict[str, torch.Tensor]`

Preprocess text and layout information for the LayoutLM model.

- `words`: List of words.
- `boxes`: List of bounding boxes for the words.
- `max_seq_length`: Maximum sequence length for the model.
- `pad_token_box`: Padding token box coordinates.
- Returns: Dictionary of tensors ready for model input.

### `embed_page(words: List[str], boxes: List[List[int]]) -> np.ndarray`

Embed a single page using the LayoutLM-Byne model.

- `words`: List of words on the page.
- `boxes`: List of bounding boxes for the words.
- Returns: Numpy array representing the page embedding.

### `process_pdf(pdf_path: str) -> Tuple[np.ndarray, List[List[str]], List[Image.Image]]`

Process a PDF file and return embeddings, extracted words, and images for each page.

- `pdf_path`: Path to the PDF file.
- Returns: A tuple containing:
  1. Numpy array of page embeddings.
  2. List of lists, where each inner list contains words extracted from a page.
  3. List of PIL Image objects, one for each page of the PDF.

### `embed_queries(queries: List[str]) -> np.ndarray`

Embed a list of queries using the LayoutLM-Byne model.

- `queries`: List of query strings.
- Returns: Numpy array of query embeddings.

### `search_relevant_pages(page_embeddings: np.ndarray, query_embeddings: np.ndarray) -> np.ndarray`

Search for relevant pages based on query embeddings.

- `page_embeddings`: Numpy array of page embeddings.
- `query_embeddings`: Numpy array of query embeddings.
- Returns: Numpy array of relevant page indices for each query.

### `answer_question(page_image: Image.Image, page_words: List[str], query: str) -> str`

Answer a question about a specific page or return raw text if MiniCPM is not loaded.

- `page_image`: PIL Image object of the page.
- `page_words`: List of words extracted from the page.
- `query`: String containing the question.
- Returns: String containing the answer or raw text of the page.

### `process_query(pdf_path: str, query: str) -> str`

Process a PDF and answer a query about it (full pipeline).

- `pdf_path`: Path to the PDF file.
- `query`: String containing the question.
- Returns: String containing the answer or raw text of the most relevant page.

## Usage Example

```python
from bynepdf import PDFQueryProcessor

# Initialize with MiniCPM (default behavior)
processor = PDFQueryProcessor(verbose=True)

# Or initialize without MiniCPM to get raw text instead of answers
# processor = PDFQueryProcessor(verbose=True, load_minicpm=False)

pdf_path = 'path/to/your/pdf'
query = "What is the main topic of this document?"
result = processor.process_query(pdf_path, query)
print(f"Result: {result}")
```

## Notes

- The LayoutLM-Byne model is always used for document and query embedding and cannot be substituted.
- When `load_minicpm` is set to False, the `answer_question` method will return the raw text of the page instead of generating an answer.
- The `verbose` parameter can be used to enable detailed processing information output.
- The library handles PDF processing, OCR, embedding, and (optionally) question answering in an end-to-end fashion.
- Make sure you have the necessary dependencies installed, including `torch`, `transformers`, `Pillow`, `pytesseract`, `pdf2image`, `numpy`, and `scikit-learn`.