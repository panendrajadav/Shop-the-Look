# 👗 Shop-the-Look: Visual Product Discovery

An end-to-end Machine Learning system that identifies garments from lifestyle outfit photos and retrieves exact catalog matches or visually similar products using **OpenCLIP (ViT-B/32)** and **FAISS Vector Search**.

---

## 🚀 Key Features

- **Whole Look & Garment Localization:** Detects full one-piece dresses (`👗 Dress`), complete outfits (`✨ Full Outfit / Look`), tops & shirts (`👚 Top / T-shirt`), jackets (`🧥 Jacket`), pants (`👖 Pants / Trousers`), skirts (`👗 Skirt`), footwear (`👠 Footwear`), and bags (`👜 Bag`).
- **Semantic Vector Search:** Uses OpenAI **OpenCLIP ViT-B/32** (512-D normalized embeddings) paired with **FAISS `IndexFlatIP`** for high-speed cosine similarity search.
- **Category & Color Reranking:** Filters out category mismatches and reranks candidates using dominant color palette and garment silhouette.
- **Honest Exact Matching:** Never classifies products as exact matches based solely on high cosine similarity. Real uploads return *Visually Similar Candidate* unless verified with ground-truth catalog evidence.
- **Interactive Web App:** Streamlit UI with live bounding box overlays, garment selector, and product cards with visual explanations.

---

## 📦 Installation Guide

### 1. Prerequisites
- Python **3.10+** (tested up to 3.13)
- Windows, macOS, or Linux

### 2. Create & Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🎯 How to Run

### 1. Launch Interactive Web Demo
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser:
- **Upload Custom Image** or **Pick from Validation Scenes**.
- Click through detected garments (Dress, Top, Pants, Footwear, etc.) to view matched catalog products.
- Use the **`🔄 Reload Cache & Index`** button in the sidebar anytime to refresh models and index in memory.

### 2. Run Validation Benchmark
```bash
python scripts/evaluate.py
```
Evaluates Recall@1, Recall@5, Recall@10, and MRR on `data/validation.jsonl`. Results are saved to `artifacts/results/evaluation_results.json`.

### 3. (Optional) Rebuild Catalog Index
The repository includes a prebuilt FAISS index of **3,011 products** (including 100% of validation ground-truth items) in `artifacts/embeddings/`. To rebuild or change the limit:
```bash
python scripts/build_index.py --limit 1500
```

---

## 🏗️ Architecture & Pipeline

```text
[Outfit Image] ──► [Garment Detector] ──► [OpenCLIP ViT-B/32] ──► [FAISS Index] ──► [Reranker & Explainer]
                     (Dress / Tops /        (512-D L2 Norm)       (3,011 items)       (Category, Color,
                      Pants / Shoes / Bags)                                            Honest Match Label)
```

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Garment Localization** | Faster R-CNN + Anatomical Cropper | Detects people/bags, isolates whole dress or separates without pants/top confusion |
| **Feature Extraction** | OpenCLIP ViT-B/32 | 512-dimensional normalized joint image-text embeddings |
| **Vector Search** | FAISS (`IndexFlatIP`) | Exact inner product (cosine similarity) vector index |
| **Attribute Engine** | HSV Clustering + Zero-Shot CLIP | Extracts dominant color, pattern, and silhouette for transparent explanations |
| **Web Interface** | Streamlit | Clean, interactive frontend for testing uploads and benchmark scenes |

---

## 📊 Dataset Summary

| File | Records | Content |
| :--- | :--- | :--- |
| `data/product_catelog.jsonl` | 50,000 rows (28,094 unique IDs) | Catalog inventory product IDs (deduplicated at index time). |
| `data/validation.jsonl` | 330 pairs (321 scenes, 206 products) | Ground-truth scene-to-product associations for evaluation. |

*Images are fetched on demand and cached locally in `artifacts/images/` using the Pinterest CDN URL formula.*

---

## 📁 Project Structure

```text
Shop-the-Look/
├── README.md                      # Short & clear project guide (this file)
├── TECHNICAL_REPORT.md            # In-depth technical analysis & metrics
├── app.py                         # Streamlit interactive application
├── config.py                      # Global configurations & file paths
├── requirements.txt               # Pinned Python package dependencies
├── data/                          # Catalog & validation JSONL datasets
├── src/                           # Core source library
│   ├── catalog_index.py           # FAISS vector index & metadata persistence
│   ├── data_loader.py             # JSONL dataset loaders
│   ├── evaluation.py              # Quantitative evaluation pipeline
│   ├── explanations.py            # Visual attribute explanation generator
│   ├── garment_detector.py        # Whole-image dress & garment localization
│   ├── image_downloader.py        # Multithreaded downloader with CDN caching
│   ├── image_encoder.py           # OpenCLIP ViT-B/32 encoder
│   └── retrieval.py               # Candidate retrieval & reranking
├── scripts/                       # Executable workflows
│   ├── build_index.py             # Index generation script
│   ├── evaluate.py                # Evaluation runner
│   └── enrich_catalog_metadata.py # Metadata enrichment utility
└── artifacts/                     # Local persisted index, embeddings & cache
```

---

## 💡 Quick Tips

- **Port in use:** If port 8501 is busy, run `streamlit run app.py --server.port 8502`.
- **PowerShell Script Policy:** If PowerShell blocks script activation, run:  
  `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`.
