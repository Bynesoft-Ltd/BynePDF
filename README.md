# BynePDF

BynePDF is a simple wrapper for our SOTA model Byne-LayoutLM. This model enables the retrieval of pages from visually rich documents. 
With this library, you can implement document-understanding systems and create visual or agentic RAG pipelines with industry-leading performance when analysing documents like pitch decks, company reports or scientific papers.

## Features

- PDF to image conversion
- OCR text extraction with bounding boxes
- Document and query embedding using LayoutLM-Byne
- Relevant page search
- Question answering using MiniCPM

## Installation

You can install BynePDF directly from GitHub using pip:

```bash
pip install git+https://github.com/Byne/BynePDF.git
```

## Dependencies

BynePDF requires the following dependencies:

- torch
- transformers
- Pillow
- pytesseract
- pdf2image
- numpy
- scikit-learn

These will be automatically installed when you install the library.

## Quick Start

```python
from bynepdf import PDFQueryProcessor

processor = PDFQueryProcessor(verbose=True)
pdf_path = 'path/to/your/pdf'
query = "What is the main topic of this document?"
answer = processor.process_query(pdf_path, query)
print(f"Answer: {answer}")
```

## Key Components

1. **LayoutLM-Byne**: This model is used for document and query embedding. It's pre-trained on a large corpus of document images and is specifically designed for document understanding tasks. In BynePDF, this model is non-substitutable and is always used for embedding.

2. **MiniCPM**: This model is used for question answering. It takes the embedded document and query as input and generates an answer. The path to this model can be customized during initialization.

## Documentation

For detailed documentation on each method, please see the [API Documentation](docs/API.md) file.

## Models

BynePDF uses the following models:

- [LayoutLM-Byne v0.1](https://huggingface.co/Byne/LayoutLM-Byne-v0.1) for document embedding (non-substitutable)
- [MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6) for question answering (default)

For more information on the LayoutLM-Byne model and its performance, check out our [blog post](https://example.com/layoutlm-byne-blog-post).

## Usage Notes

- The LayoutLM-Byne model is always used for document and query embedding and cannot be substituted.
- The MiniCPM model path can be customized during initialization if needed.
- The library handles PDF processing, OCR, embedding, and question answering in an end-to-end fashion.
- Verbose mode can be enabled for detailed processing information.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues or have questions, please file an issue on the GitHub repository.