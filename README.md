# Shop-the-Look: Visual Product Discovery

An AI-powered fashion discovery application that identifies clothing and accessories in real-world images and retrieves matching or visually similar products from a catalog.

## Features

- **Fashion Item Detection:** Identifies clothing items and accessories in scene images.
- **Visual Product Retrieval:** Finds visually similar products from the catalog using image embeddings.
- **Similarity Ranking:** Ranks candidate products based on visual similarity and relevant attributes.
- **Visual Explanations:** Highlights similarities such as color, pattern, category, and appearance.
- **Interactive Demo:** Explore product recommendations through a Streamlit interface.

## Tech Stack

- Python
- OpenCLIP (ViT-B/32)
- FAISS
- Faster R-CNN
- PyTorch
- Streamlit
- NumPy and Pillow

## Project Structure

```text
shop-the-look/
├── app.py
├── config.py
├── requirements.txt
├── data/
│   ├── product_catelog.jsonl
│   └── validation.jsonl
├── src/
│   ├── data_loader.py
│   ├── image_downloader.py
│   ├── image_encoder.py
│   ├── catalog_index.py
│   ├── garment_detector.py
│   ├── retrieval.py
│   ├── explanations.py
│   └── evaluation.py
├── scripts/
│   ├── build_index.py
│   └── evaluate.py
└── artifacts/
```

## Setup and Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Build a Small Index

```bash
python scripts/build_index.py --limit 1000
```

### 3. Build the Full Catalog Index

```bash
python scripts/build_index.py --limit 0
```

### 4. Run Evaluation

```bash
python scripts/evaluate.py
```

### 5. Launch the Application

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## How It Works

1. Upload or select a fashion scene image.
2. Detect clothing items and accessories.
3. Generate visual embeddings for detected items.
4. Retrieve and rank similar products from the catalog.
5. View recommended products and their visual similarities.

## Limitations and Future Improvements

- Improve garment detection for complex scenes and overlapping items.
- Enhance category classification for visually similar clothing types.
- Improve retrieval quality through fine-tuning and better ranking strategies.
- Incorporate richer product metadata when available.

## License

Add the appropriate license if the project is released as open source.
