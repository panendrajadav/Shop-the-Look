# Shop-the-Look: Visual Product Discovery

## Project overview

Shop-the-Look takes a real-world fashion photo, finds the individual items in it (jacket, shirt, trousers, skirt, dress, shoes, bag, sunglasses) and looks each one up in a product catalog.

- If the photo is itself a catalog image, the product is returned as an **exact match**.
- Otherwise the **most visually similar** catalog products are returned for each item, with an explanation of what they share with the item and where they differ (category, color, pattern, style, material, overall appearance).

The system is zero-shot: it uses pretrained models only and nothing is trained on this dataset.

| Part | Technology |
| :--- | :--- |
| Item detection | Faster R-CNN (MobileNetV3, COCO) person and bag boxes, body-region crops, zero-shot CLIP classification and verification |
| Embeddings | OpenCLIP ViT-B/32 (OpenAI weights), 512-D, L2-normalized |
| Search | FAISS `IndexFlatIP`, category-restricted, domain-centred scores |
| Demo | Streamlit |
| Hardware | CPU only |

| Deliverable | Where |
| :--- | :--- |
| Source code | `src/`, `scripts/`, `config.py` |
| Demo interface | `app.py` |
| Technical document | `TECHNICAL_REPORT.md` |

### Data

| Item | Value |
| :--- | :--- |
| `data/product_catelog.jsonl` | 50,000 rows, 28,094 unique product IDs |
| `data/validation.jsonl` | 330 scene-product pairs, 321 scenes, 206 products (165 of them in the catalog) |
| Catalog images downloaded | 22,213 |
| Products embedded and indexed | 8,762 |
| Scene images downloaded | 321 of 321 |

The files contain IDs only. Images are fetched from the Pinterest CDN and cached under `artifacts/images/`.

Only part of the catalog is indexed so far. Encoding runs at about 8 images/s on the CPU-only development laptop, so the build was time-boxed. It is resumable: running the build again downloads the missing images and embeds everything not yet indexed.

## Setup

Requirements: Python 3.10 or newer (developed on 3.13), about 2 GB of free disk space for images and embeddings, and an internet connection for the first run (model weights and catalog images).

**1. Create and activate a virtual environment**

Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Build the catalog index**

`artifacts/` is git-ignored, so a fresh clone has no images or embeddings and the index must be built once:

```bash
python scripts/build_index.py
```

This runs every build step in order:

1. Downloads catalog and scene images (cached; already-downloaded files are skipped).
2. Embeds catalog images with OpenCLIP and writes the FAISS index.
3. Estimates item name, category, pattern and color for each product.
4. Adds upper-body and lower-body views for tops, jackets, pants and skirts.
5. Computes the query-domain mean used for domain centring (from scene images only, no labels).

Useful options:

```bash
python scripts/build_index.py --limit 5000                      # index at most 5,000 products
python scripts/build_index.py --cached-only --max-minutes 10    # no downloads, stop encoding after 10 minutes
```

The full catalog takes several hours on a laptop CPU. A partial build works: the demo and evaluation use whatever is indexed.

## Running the application

**Start the demo**

```bash
streamlit run app.py
```

Then open http://localhost:8501. If the port is busy, add `--server.port 8502`.

**Use the demo**

1. Choose **Upload Custom Image** and upload a photo, or choose **Pick from Validation Scenes** and select one of the 321 scenes.
2. The photo is shown with a box and name for each detected item.
3. **Full Outfit** lists the single best catalog match for every detected item, labelled as an exact match or the most similar product.
4. Select one item to see more candidates for it. Each card shows the similarity score, the product's attributes, and what it shares with and differs from the item in the photo.
5. For validation scenes, open **Benchmark answer** to see the labelled product and whether it was retrieved. This panel is display-only and does not influence the results.
6. After rebuilding the index, press **Reload Cache & Index** in the sidebar.

To see an exact match, upload any catalog image from `artifacts/images/`.

**Run the evaluation**

```bash
python scripts/evaluate.py                   # detect, match and score all validation scenes
python scripts/evaluate.py --reuse-queries   # reuse cached detections; only valid if detection code is unchanged
```

Outputs in `artifacts/results/`:

| File | Content |
| :--- | :--- |
| `evaluation_results.json` | Recall@K, MRR and the stage-by-stage breakdown |
| `per_scene_evaluation.csv` | Per-scene recall and top retrieved products |
| `scene_breakdown.csv` | Per scene: detected items, whether the labelled product is indexed, whether its category was detected, and its rank within the category, after reranking and overall |

**Re-run a single build step** (after changing prompts, categories or detection code):

```bash
python scripts/enrich_catalog_metadata.py   # item name, category, pattern, color histograms
python scripts/build_product_views.py       # upper/lower garment views (incremental)
python scripts/build_query_stats.py         # query-domain mean and cached scene detections
```

## How it works

1. **Detect items** (`src/garment_detector.py`)
   - Faster R-CNN finds the largest person and any bag.
   - The person box is split by body proportions into outer layer, inner shirt (when the outer layer is a jacket), lower body, footwear and head regions.
   - Each region is classified with zero-shot CLIP prompts and given a specific name such as Jeans, Denim Skirt, Blazer or Sneakers.
   - Each region is verified against all item categories plus background prompts and dropped if it does not show that garment. A shoe close-up yields shoes, not an invented coat and trousers.
   - If no person is found, or nothing on the person passes verification, the whole image is treated as one item.
2. **Embed** (`src/image_encoder.py`, `src/retrieval.py`)
   - A detected item is embedded as the average of its crop, the mirrored crop and the central 80% of the crop.
   - Catalog tops, jackets, pants and skirts are represented by their whole image plus an upper-body or lower-body view, because many catalog photos show a model wearing the product with other clothes.
3. **Search** (`src/catalog_index.py`)
   - Candidates are restricted to the item's category: a shirt query returns shirts. A product whose photo is ambiguous (a model in a tee and jeans) is also reachable through its second most likely category.
   - Products are ordered by domain-centred similarity: the average scene-crop vector is subtracted from the query and the average catalog vector from the products before comparing, which removes the "street photo" versus "product photo" difference and leaves the garment-specific part.
4. **Rerank** (`src/retrieval.py`, `src/color_features.py`)
   - Small adjustments for color overlap (28-bin color histograms), pattern agreement and same specific item.
5. **Exact or similar** (`src/explanations.py`)
   - A high similarity score only nominates a candidate. An exact match is claimed only when the query and the catalog image also agree pixel by pixel, which is true for the same photo and false for a different photo of even the same product.
   - Everything else is labelled "Visually Similar".
6. **Explain**
   - Each product card lists what it shares with the item and where it differs on category, color, pattern, style and material, plus the similarity score.

## Results on `validation.jsonl`

282 of the 321 scenes have their labelled product in the catalog and are scored. The other 39 point to products that are not in the catalog file, so only similar items can be returned for them. Scoring is against the 8,762 indexed products.

Last completed full run (before domain centring and score fusion were added):

| Metric | Value |
| :--- | :--- |
| Recall@1 | 0.014 |
| Recall@5 | 0.028 |
| Recall@10 | 0.036 |
| Recall@20 | 0.064 |
| MRR | 0.024 |
| Labelled product in the demo's top 5 for some detected item | 0.060 |

Where the labelled product was lost in that run:

- The detector found the labelled item's category in 182 of 282 scenes and missed it in 100, most often bags (32), footwear (17), pants (16) and accessories other than sunglasses (16).
- When the category was found, the labelled product's median rank within its category was 256, and it was in the category's top 10 in 15 scenes.

Domain centring and score fusion were added in response. In an offline test on the same detections they put the labelled product in the top 10 for 21 of 282 scenes (0.074) and the top 20 for 30 (0.106). The full pipeline run with these changes has not finished yet; run `python scripts/evaluate.py` for the current figures.

**No validation leakage.** The index is built from the catalog file only, and the build asserts that every indexed ID is a catalog ID. Detection, ranking and the exact/similar decision never see the labels; the evaluator reads them only to score finished predictions. The query-domain mean is computed from scene images without labels. Rerank weights were chosen with an ablation on these same scenes and kept as small as possible; that is a use of the validation set for model selection and is reported as such.

## Project structure

```text
app.py                          Streamlit demo
config.py                       Paths and settings
requirements.txt
TECHNICAL_REPORT.md             Technical document
data/                           product_catelog.jsonl, validation.jsonl
src/
  data_loader.py                JSONL loaders
  image_downloader.py           Parallel, cached image download
  image_encoder.py              OpenCLIP encoder
  catalog_index.py              FAISS index, metadata, category-restricted and centred search
  garment_detector.py           Item detection, naming and verification
  color_features.py             Color histograms for reranking
  retrieval.py                  Query embedding, matching and rerank
  explanations.py               Attribute comparison and exact/similar decision
  evaluation.py                 Recall@K, MRR, per-scene breakdown
scripts/
  build_index.py                Full build: download, embed, index, then the three steps below
  enrich_catalog_metadata.py    Item name, category, pattern, color per product
  build_product_views.py        Upper/lower garment views of catalog photos
  build_query_stats.py          Query-domain mean and cached scene detections
  evaluate.py                   Validation benchmark
artifacts/                      Images, embeddings, results (git-ignored)
```

## Known limitations

- **The exact product is rarely ranked first from a scene photo.** See the results above. The encoder is general-purpose and the catalog contains many near-identical products. A fashion-tuned encoder or a model trained on scene-product pairs would be needed to change this substantially.
- **Detection uses body proportions, not a clothing detector.** Crops can include neighbouring items or background, only the largest person is analysed, and belts, hats and jewellery are not detected (sunglasses are). A dress is sometimes split into top and skirt, and items outside the frame cannot be found.
- **Attributes are estimates.** The catalog provides IDs only, so item name, category, color, pattern, style and material come from the images and can be wrong. A product filed under the wrong category appears under the wrong item.
- **Partial coverage.** 8,762 of 28,094 unique products are indexed until the build is run to completion. Indexing the rest improves coverage; it does not by itself improve ranking accuracy.
