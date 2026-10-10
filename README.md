# Shop-the-Look: Visual Product Discovery

Given a real-world outfit photo, this system finds the fashion items in it and matches each one against the product catalog. If the catalog holds the same product image it is flagged as a likely exact match; otherwise the most visually similar products are returned, each with an explanation covering category, color, pattern, style, material and overall appearance.

It is built on OpenCLIP ViT-B/32 embeddings, a FAISS index and a Faster R-CNN person detector, and runs on CPU. No model is trained on this dataset.

| Deliverable | Where |
| :--- | :--- |
| Source code | `src/`, `scripts/`, `config.py` |
| Demo interface | `app.py` (Streamlit) |
| Technical document | `TECHNICAL_REPORT.md` |

## Current state

| Item | Value |
| :--- | :--- |
| Catalog file | 50,000 rows, 28,094 unique product IDs |
| Catalog images downloaded | 22,213 |
| Products embedded and indexed | 8,762 |
| Garment views of catalog photos | 9,268 (upper and lower crops of 4,634 tops, jackets, pants and skirts) |
| Validation scenes downloaded | 321 of 321 |
| Encoder | OpenCLIP ViT-B/32 (OpenAI weights), CPU |

Only part of the catalog is indexed. Encoding runs at about 8 images/s on the CPU-only development laptop, so the build was time-boxed. The build is resumable: `python scripts/build_index.py` downloads the missing images and embeds everything not yet indexed.

## Results on `validation.jsonl`

Measured with `python scripts/evaluate.py` against the 8,762-product index. 282 of the 321 scenes have their labelled product in the catalog and are scored. The other 39 point to products that are not in the catalog file, so only similar items can be returned for them.

| Metric | Value |
| :--- | :--- |
| Recall@1 | 0.018 |
| Recall@5 | 0.036 |
| Recall@10 | 0.050 |
| Recall@20 | 0.071 |
| MRR | 0.029 |
| Labelled product in the demo's top 3 for some detected item | 0.046 |
| Labelled product in the demo's top 5 for some detected item | 0.050 |
| "Likely exact match" badges fired on scenes | 0 |

These numbers were measured before the latest reranking changes (color histograms, mirrored and centre-cropped query embeddings, tighter garment regions, second-category handling). Those changes have not been benchmarked yet; re-run `python scripts/evaluate.py` for current figures.

**How to read this.** Retrieval of the exact labelled product from a scene photo is weak: the labelled product is in the top 10 for about 1 scene in 20. A general-purpose encoder does not place a street photo of a worn garment close to a catalog photo of the same product, and the catalog has many near-identical items. What the system does reliably is return products of the right type and similar look, and recognise a catalog image when it is given one:

- 30 of 30 catalog images submitted as queries came back as rank 1 with the "Likely Exact Match" badge. This was last run before the latest reranking changes.
- Stored embeddings were verified against their images: all 8,762 indexed IDs are unique catalog products, every image path matches its ID, and re-embedding 24 sampled products reproduced the stored vectors exactly.

### No validation leakage

- The index is built from `data/product_catelog.jsonl` only. `scripts/build_index.py` never reads validation labels and asserts that every indexed ID is a catalog ID.
- Detection, ranking and the exact/similar decision never see the labels. They are read only by the evaluator, to score finished predictions, and by the demo's "Benchmark answer" panel, which is display-only.

## How it works

1. **Detect items** (`src/garment_detector.py`)
   - Faster R-CNN (MobileNetV3, COCO weights) finds the largest person and any bag.
   - The person box is split by body proportions into separate regions with little overlap: outer layer, inner shirt (when the outer layer is a jacket), lower body, footwear and head.
   - Each region is classified with zero-shot CLIP prompts into top, jacket, pants, skirt, dress, footwear, bag or sunglasses, and given a specific name such as Jeans, Denim Skirt, Blazer or Sneakers.
   - Each region is then verified against all item categories plus background prompts and dropped if it does not show that garment. A shoe close-up therefore yields shoes, not an invented coat and trousers.
   - If no person is found, or nothing on the person passes verification, the whole image is treated as one item.
2. **Embed** (`src/image_encoder.py`, `src/retrieval.py`)
   - Catalog images become normalized 512-D OpenCLIP vectors.
   - A detected item is embedded as the average of its crop, the mirrored crop and the central 80% of the crop, which reduces the background picked up at region edges.
   - Catalog tops, jackets, pants and skirts also get an upper-body and a lower-body view (`scripts/build_product_views.py`), because many catalog photos show a model wearing the product with other clothes. A product is scored by the better of its whole image and the view matching its category.
3. **Search** (`src/catalog_index.py`)
   - Candidates are restricted to the item's category: a shirt query returns shirts, a sunglasses query returns sunglasses.
   - A product whose photo is ambiguous (a model in a tee and jeans) is also reachable through its second most likely category.
   - A product outside the category enters only if it is a near-duplicate of the query (cosine 0.97 or more), so an exact image is never lost to a wrong category estimate.
4. **Rerank** (`src/retrieval.py`, `src/color_features.py`)
   - Score = cosine similarity, plus adjustments for category agreement, color overlap (intersection of 28-bin color histograms of the garment pixels), pattern agreement and same specific item.
5. **Exact or similar** (`src/explanations.py`)
   - The top result is labelled "Likely Exact Match" when its similarity is at least 0.93 and it leads the runner-up by 0.03 or more; 0.985 or above counts as the same image outright.
   - The whole photo is also checked against the index, so a catalog image is recognised even when no garment crop covers the full frame.
   - Everything else is labelled "Visually Similar".
6. **Explain**
   - Each product card lists what it shares with the query and where it differs on category, color, pattern, style and material, plus the similarity score.

## Run it

```bash
pip install -r requirements.txt

# Demo
streamlit run app.py

# Evaluate on validation.jsonl (writes artifacts/results/)
python scripts/evaluate.py

# Build or extend the index (downloads missing images, resumes from existing embeddings)
python scripts/build_index.py
python scripts/build_index.py --cached-only --max-minutes 10   # no downloads, time-boxed
```

`build_index.py` also runs the two follow-up steps, which can be run on their own after changing prompts or categories:

```bash
python scripts/enrich_catalog_metadata.py   # item name, category, pattern, color histograms
python scripts/build_product_views.py       # upper/lower garment views (incremental)
```

### Using the demo

- Upload a photo or pick any of the 321 validation scenes.
- **Full Outfit** shows the single best catalog match for every detected item, labelled as an exact match or the most similar product.
- Selecting one item shows more candidates for it, with the explanation for each.
- For validation scenes, the "Benchmark answer" panel shows the labelled product and whether it was retrieved.
- To check exact matching, upload any catalog image from `artifacts/images/`.

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
  catalog_index.py              FAISS index, metadata, category-restricted search
  garment_detector.py           Item detection, naming and verification
  color_features.py             Color histograms for reranking
  retrieval.py                  Query embedding and rerank
  explanations.py               Attribute comparison and exact/similar decision
  evaluation.py                 Recall@K, MRR, demo hit rate
scripts/
  build_index.py                Download, embed, index
  enrich_catalog_metadata.py    Item name, category, pattern, color per product
  build_product_views.py        Upper/lower garment views of catalog photos
  evaluate.py                   Validation benchmark
artifacts/                      Images, embeddings, results (git-ignored)
```

## Known limitations

- **Exact product from a scene photo is rarely found.** See the results above. A fashion-tuned encoder or a model trained on scene-product pairs would be needed to change this.
- **Regions come from body proportions, not a clothing detector.** Crops can include neighbouring items or background, only the largest person is analysed, and belts, hats and jewellery are not detected (sunglasses are).
- **Known detection mistakes.** A dress is sometimes split into top and skirt, and shorts or boots are sometimes missed.
- **Attributes are estimates.** The catalog provides IDs only, so item name, category, color, pattern, style and material come from the images and can be wrong. A product filed under the wrong category appears under the wrong item.
- **Partial coverage.** 8,762 of 28,094 unique products are indexed until the build is run to completion. Indexing the rest improves coverage, not accuracy.
