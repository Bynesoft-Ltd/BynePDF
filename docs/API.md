# BynePDF API Documentation

## PDFQueryProcessor

The main class for processing PDFs, embedding their content, and answering queries.

### `__init__(minicpm_path: str = 'openbmb/MiniCPM-V-2_6', device: Optional[str] = None, verbose: bool = False)`

Initialize the PDFQueryProcessor.

- `minicpm_path`: Path to the MiniCPM model. Defaults to 'openbmb/MiniCPM-V-2_6'.
- `device`: Device to use for computation ('cuda' or 'cpu'). If None, will use CUDA if available.
- `verbose`: Whether to print verbose output.

Note: The LayoutLM-Byne model is always used and cannot be substituted.

### `split_pdf_to_images(pdf_path: str) -> List[Image.Image]`

Convert a PDF file to a list of images.

- `pdf_path`: Path to the PDF file.
- Returns: List of PIL Image objects, one for each page of the PDF.

### `apply_tesseract(image: Image.Image) -> Tuple[List[str], List[List[int]]]`

Apply OCR to an image and return words and bounding boxes.

- `image`: PIL Image object to process.
- Returns: A tuple containing a list of words and a list of normalized bounding boxes.

### `embed_page(words: List[str], boxes: List[List[int]]) -> np.ndarray`

Embed a single page using the LayoutLM-Byne model.

- `words`: List of words on the page.
- `boxes`: List of bounding boxes for the words.
- Returns: Numpy array representing the page embedding.

### `process_pdf(pdf_path: str) -> np.ndarray`

Process a PDF file and return embeddings for each page.

- `pdf_path`: Path to the PDF file.
- Returns: Numpy array of page embeddings.

### `embed_queries(queries: List[str]) -> np.ndarray`

Embed a list of queries using the LayoutLM-Byne model.

- `queries`: List of query strings.
- Returns: Numpy array of query embeddings.

### `search_relevant_pages(page_embeddings: np.ndarray, query_embeddings: np.ndarray) -> np.ndarray`

Search for relevant pages based on query embeddings.

- `page_embeddings`: Numpy array of page embeddings.
- `query_embeddings`: Numpy array of query embeddings.
- Returns: Numpy array of relevant page indices for each query.

### `answer_question(page_image: Image.Image, query: str) -> str`

Answer a question about a specific page image using MiniCPM.

- `page_image`: PIL Image object of the page.
- `query`: String containing the question.
- Returns: String containing the answer.

### `process_query(pdf_path: str, query: str) -> str`

Process a PDF and answer a query about it (full pipeline).

- `pdf_path`: Path to the PDF file.
- `query`: String containing the question.
- Returns: String containing the answer.

## Usage Example

```python
from bynepdf import PDFQueryProcessor

processor = PDFQueryProcessor(verbose=True)
pdf_path = 'path/to/your/pdf'
query = "What is the main topic of this document?"
answer = processor.process_query(pdf_path, query)
print(f"Answer: {answer}")
```

## Notes

- The LayoutLM-Byne model (Byne/LayoutLM-Byne-v0.1) is used by default for document and query embedding and cannot be substituted.
- The MiniCPM model is used for question answering and can be optionally specified during initialization.
- All methods that involve model inference (embed_page, process_pdf, embed_queries, answer_question) use the appropriate model (LayoutLM-Byne or MiniCPM) internally.
- The library handles PDF to image conversion, OCR, embedding, and question answering in an end-to-end fashion.