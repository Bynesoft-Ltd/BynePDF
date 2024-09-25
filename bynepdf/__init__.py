import torch
from transformers import LayoutLMForSequenceClassification, AutoTokenizer, AutoModel
from PIL import Image
import pytesseract
from typing import List, Optional, Tuple, Union, Dict
from pdf2image import convert_from_path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings

BYNE_LAYOUTLM_PATH = 'Byne/LayoutLM-Byne-v0.1'


class PDFQueryProcessor:
    def __init__(self, minicpm_path: str = 'openbmb/MiniCPM-V-2_6', device: Optional[str] = None,
                 verbose: bool = False, load_minicpm: bool = True):
        self.verbose = verbose
        self.device = self._get_device(device)

        # LayoutLM-Byne setup (non-substitutable)
        self.layoutlm_tokenizer = AutoTokenizer.from_pretrained(BYNE_LAYOUTLM_PATH)
        self.layoutlm_model = LayoutLMForSequenceClassification.from_pretrained(BYNE_LAYOUTLM_PATH, num_labels=768).to(
            self.device).eval()

        # MiniCPM setup
        self.minicpm_tokenizer = None
        self.minicpm_model = None
        self.load_minicpm = load_minicpm

        if load_minicpm:
            self.minicpm_tokenizer = AutoTokenizer.from_pretrained(minicpm_path, trust_remote_code=True)
            self.minicpm_model = AutoModel.from_pretrained(minicpm_path, trust_remote_code=True,
                                                           attn_implementation='sdpa', torch_dtype=torch.bfloat16).to(
                self.device).eval()
        else:
            warnings.warn("MiniCPM is not loaded. The pipeline will return raw text for relevant pages instead of answering questions.")

    def _get_device(self, device: Optional[str]) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device not in ["cuda", "cpu"]:
            raise ValueError("Device must be either 'cuda' or 'cpu'")
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device)

    def split_pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        try:
            images = convert_from_path(pdf_path)
            if self.verbose:
                print(f"Converted PDF to {len(images)} images")
            return images
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF to images: {str(e)}")

    @staticmethod
    def normalize_box(box: List[int], width: int, height: int) -> List[int]:
        return [
            int(1000 * (box[0] / width)),
            int(1000 * (box[1] / height)),
            int(1000 * (box[2] / width)),
            int(1000 * (box[3] / height)),
        ]

    def apply_tesseract(self, image: Image.Image) -> Tuple[List[str], List[List[int]]]:
        data = pytesseract.image_to_data(image, lang=None, output_type="dict", config="")
        words, left, top, width, height = data["text"], data["left"], data["top"], data["width"], data["height"]

        valid_indices = [idx for idx, word in enumerate(words) if word.strip()]
        words = [words[idx] for idx in valid_indices]
        boxes = [[left[idx], top[idx], left[idx] + width[idx], top[idx] + height[idx]] for idx in valid_indices]

        image_width, image_height = image.size
        normalized_boxes = [self.normalize_box(box, image_width, image_height) for box in boxes]

        if self.verbose:
            print(f"OCR extracted {len(words)} words from image")
        return words, normalized_boxes

    def preprocess_for_model(self, words: List[str], boxes: List[List[int]],
                             max_seq_length: int = 512,
                             pad_token_box: List[int] = [0, 0, 0, 0]) -> Dict[str, torch.Tensor]:
        assert len(words) == len(boxes), "Number of words must match number of boxes"

        token_boxes = []
        for word, box in zip(words, boxes):
            word_tokens = self.layoutlm_tokenizer.tokenize(word)
            token_boxes.extend([box] * len(word_tokens))

        special_tokens_count = 2
        if len(token_boxes) > max_seq_length - special_tokens_count:
            token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]

        token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

        encoding = self.layoutlm_tokenizer(' '.join(words), padding='max_length', truncation=True,
                                           max_length=max_seq_length, return_tensors="pt")

        input_ids = self.layoutlm_tokenizer(' '.join(words), truncation=True, max_length=max_seq_length)["input_ids"]
        padding_length = max_seq_length - len(input_ids)
        token_boxes += [pad_token_box] * padding_length

        encoding['bbox'] = torch.tensor([token_boxes])
        encoding.pop('token_type_ids', None)

        return {k: v.to(self.device) for k, v in encoding.items()}

    def embed_page(self, words: List[str], boxes: List[List[int]]) -> np.ndarray:
        encoding = self.preprocess_for_model(words, boxes)

        with torch.no_grad():
            outputs = self.layoutlm_model(**encoding)

        if self.verbose:
            print(f"Embedded page with {len(words)} words")
        return outputs.logits.cpu().numpy().squeeze()

    def process_pdf(self, pdf_path: str) -> Tuple[np.ndarray, List[List[str]], List[Image.Image]]:
        images = self.split_pdf_to_images(pdf_path)
        page_embeddings = []
        all_words = []

        for i, image in enumerate(images):
            words, boxes = self.apply_tesseract(image)
            embedding = self.embed_page(words, boxes)
            page_embeddings.append(embedding)
            all_words.append(words)
            if self.verbose:
                print(f"Processed page {i + 1}/{len(images)}")

        return np.array(page_embeddings).squeeze(), all_words, images

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        query_embeddings = []
        for i, query in enumerate(queries):
            words = query.split()
            boxes = [[118, 47, 185, 57]] * len(words)  # Use dummy boxes for queries
            embedding = self.embed_page(words, boxes)
            query_embeddings.append(embedding)
            if self.verbose:
                print(f"Embedded query {i + 1}/{len(queries)}")

        return np.array(query_embeddings)

    def search_relevant_pages(self, page_embeddings: np.ndarray,
                              query_embeddings: np.ndarray) -> np.ndarray:
        similarities = cosine_similarity(query_embeddings, page_embeddings)
        relevant_page_indices = np.argsort(-similarities, axis=1)
        if self.verbose:
            print(f"Computed relevance for {len(query_embeddings)} queries and {len(page_embeddings)} pages")
        return relevant_page_indices

    def answer_question(self, page_image: Image.Image, page_words: List[str], query: str) -> str:
        if not self.load_minicpm:
            return ' '.join(page_words)

        image = page_image.convert('RGB')
        prompt = f"Question: {query}\n\nAnswer:"
        msgs = [{'role': 'user', 'content': [image, prompt]}]

        with torch.no_grad():
            res = self.minicpm_model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.minicpm_tokenizer
            )

        return res

    def process_query(self, pdf_path: str, query: str) -> str:
        # Process PDF
        page_embeddings, all_words, images = self.process_pdf(pdf_path)

        # Embed query
        query_embedding = self.embed_queries([query])

        # Search for relevant page
        relevant_page_index = self.search_relevant_pages(page_embeddings, query_embedding)[0][0]

        # Get the relevant page image and words
        relevant_page_image = images[relevant_page_index]
        relevant_page_words = all_words[relevant_page_index]

        # Answer the question using MiniCPM or return raw text
        return self.answer_question(relevant_page_image, relevant_page_words, query)

# Usage example:
# processor = PDFQueryProcessor(verbose=True, load_minicpm=False)  # Set load_minicpm to False to skip loading MiniCPM
# pdf_path = 'path/to/your/pdf'
# query = "What is the main topic of this document?"
# answer = processor.process_query(pdf_path, query)
# print(f"Answer: {answer}")