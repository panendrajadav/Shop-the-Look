#!/usr/bin/env python3
"""Shop-the-Look: Interactive Visual Product Discovery Streamlit Application."""

import sys
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict
import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd

# Add repository root to path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from config import (
    PRODUCTS_METADATA_FILE, PRODUCTS_EMBEDDINGS_FILE, PRODUCTS_INDEX_FILE,
    VALIDATION_FILE, IMAGES_DIR, MODEL_NAME, PRETRAINED, DEVICE,
    DEFAULT_TOP_K, MAX_TOP_K, EVAL_RESULTS_JSON, EVAL_RESULTS_CSV
)
from src.image_downloader import ImageDownloader
from src.image_encoder import CLIPImageEncoder
from src.catalog_index import ProductIndex
from src.retrieval import SceneRetrieval
from src.garment_detector import GarmentDetector
from src.explanations import RecommendationExplainer
from src.data_loader import ValidationLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Shop-the-Look | Visual Discovery",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom modern CSS styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .product-card {
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        padding: 16px;
        background: #ffffff;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    .badge-exact {
        background-color: #10b981;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-similar {
        background-color: #3b82f6;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .tag {
        background-color: #f1f5f9;
        color: #475569;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing OpenCLIP and FAISS Catalog...")
def load_system():
    """Load and cache CLIP encoder, FAISS index, retrieval, and explanation engines."""
    try:
        index = ProductIndex(PRODUCTS_EMBEDDINGS_FILE, PRODUCTS_METADATA_FILE, PRODUCTS_INDEX_FILE)
        if not index.load():
            return None, None, None, None, None

        encoder = CLIPImageEncoder(model_name=MODEL_NAME, pretrained=PRETRAINED, device=DEVICE)
        downloader = ImageDownloader(IMAGES_DIR)
        garment_detector = GarmentDetector(clip_encoder=encoder, use_torchvision_detector=False)
        retrieval = SceneRetrieval(encoder=encoder, catalog_index=index, garment_detector=garment_detector)
        explainer = RecommendationExplainer(catalog_index=index, clip_encoder=encoder)

        return index, encoder, downloader, retrieval, explainer
    except Exception as e:
        logger.error(f"Error loading system: {e}")
        return None, None, None, None, None


@st.cache_data
def load_sample_scenes():
    """Load sample scene IDs from validation set."""
    try:
        val_loader = ValidationLoader(VALIDATION_FILE)
        val_loader.load()
        scenes = list(val_loader.get_all_scenes())
        return scenes, val_loader.scene_to_products
    except Exception:
        return [], {}


def main():
    st.title("🛍️ Shop-the-Look: Visual Product Discovery")
    st.caption("AI-powered outfit and product discovery from lifestyle scene images using OpenCLIP & FAISS.")

    # Load core models
    index, encoder, downloader, retrieval, explainer = load_system()

    if index is None:
        st.error(
            "⚠️ Catalog index not found or incomplete! "
            "Please build the catalog index first by running: `python scripts/build_index.py`"
        )
        st.info("The application requires precomputed embeddings and FAISS index in `artifacts/embeddings/`.")
        return

    sample_scenes, scene_to_gt = load_sample_scenes()

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        mode = st.radio(
            "Discovery Mode",
            ["Full-Scene Baseline", "Garment-Level Breakdown"],
            help="Full-scene embeds the whole image. Garment-level detects and isolates individual clothing items."
        )

        top_k = st.slider("Recommendations per item (Top-K)", min_value=1, max_value=MAX_TOP_K, value=DEFAULT_TOP_K)
        deduplicate = st.checkbox("Deduplicate recommendations", value=True)

        st.markdown("---")
        st.header("📊 Index Status")
        st.metric("Catalog Products Indexed", f"{index.get_size():,}")
        st.metric("Visual Embedding Dim", f"{encoder.get_embedding_dim()} (ViT-B-32)")
        st.metric("Inference Hardware", DEVICE.upper())

    # Tabs
    tab_discovery, tab_evaluation, tab_docs = st.tabs([
        "🔍 Visual Discovery", "📈 Benchmark & Metrics", "📑 System Architecture"
    ])

    with tab_discovery:
        col_input, col_results = st.columns([1, 1.2], gap="large")

        with col_input:
            st.subheader("1. Input Scene")
            input_source = st.radio(
                "Choose Image Source:",
                ["Pick from Validation Scenes", "Upload Custom Image"],
                horizontal=True
            )

            query_image: Optional[Image.Image] = None
            current_scene_id = None
            known_gt_ids = set()

            if input_source == "Pick from Validation Scenes":
                if sample_scenes:
                    selected_scene_idx = st.selectbox(
                        "Select a benchmark scene:",
                        options=range(min(len(sample_scenes), 50)),
                        format_func=lambda i: f"Scene #{i+1} ({sample_scenes[i][:8]}...)"
                    )
                    current_scene_id = sample_scenes[selected_scene_idx]
                    known_gt_ids = set(scene_to_gt.get(current_scene_id, []))

                    cached_path = downloader.get_cached_image(current_scene_id)
                    if cached_path and cached_path.exists():
                        try:
                            query_image = Image.open(cached_path).convert("RGB")
                        except Exception as e:
                            st.error(f"Error opening image: {e}")
                    else:
                        with st.spinner("Retrieving scene image from Pinterest CDN..."):
                            dl_path = downloader.download_image(current_scene_id)
                            if dl_path and dl_path.exists():
                                query_image = Image.open(dl_path).convert("RGB")
                            else:
                                st.warning("Could not download scene image. Please try another scene.")
                else:
                    st.info("Validation scenes not found.")

            else:
                uploaded_file = st.file_uploader(
                    "Upload a fashion or outfit photo (JPG, PNG)",
                    type=["jpg", "jpeg", "png"]
                )
                if uploaded_file:
                    try:
                        query_image = Image.open(uploaded_file).convert("RGB")
                    except Exception as e:
                        st.error(f"Invalid image file: {e}")

            if query_image is not None:
                st.image(query_image, caption="Query Scene Image", use_container_width=True)
                if current_scene_id:
                    st.caption(f"**Scene ID:** `{current_scene_id}`")
                    if known_gt_ids:
                        st.caption(f"**Known Positive Products in Scene:** {len(known_gt_ids)}")

        with col_results:
            st.subheader("2. Recommended Catalog Products")

            if query_image is None:
                st.info("👈 Select or upload a scene image to discover matching catalog products.")
            else:
                if mode == "Full-Scene Baseline":
                    with st.spinner("Searching catalog index..."):
                        results = retrieval.retrieve_from_image(
                            query_image, top_k=top_k, remove_duplicates=deduplicate
                        )

                    if not results:
                        st.warning("No matches found in the catalog.")
                    else:
                        st.success(f"Retrieved Top-{len(results)} visually similar products.")

                        explanations = explainer.explain_batch(
                            results=results,
                            query_image=query_image,
                            ground_truth_ids=known_gt_ids
                        )

                        for exp in explanations:
                            render_product_card(exp)

                else:
                    # Garment-Level Breakdown Mode
                    with st.spinner("Detecting garment regions & retrieving matching items..."):
                        garment_results = retrieval.retrieve_garments(
                            scene_image=query_image,
                            top_k_per_garment=top_k,
                            remove_duplicates=deduplicate
                        )

                    if not garment_results:
                        st.warning("No garment regions could be isolated.")
                    else:
                        st.success(f"Identified {len(garment_results)} distinct garment regions.")

                        for g_idx, g_item in enumerate(garment_results, 1):
                            with st.expander(f"👕 Item #{g_idx}: {g_item['garment_label']} ({g_item['region']})", expanded=(g_idx==1)):
                                st.image(g_item["crop_image"], width=160, caption=f"Crop: {g_item['garment_label']} ({g_item['confidence']:.1%})")
                                g_recs = g_item["recommendations"]
                                if not g_recs:
                                    st.write("No catalog items matched this crop.")
                                else:
                                    g_explanations = explainer.explain_batch(
                                        results=g_recs,
                                        query_image=g_item["crop_image"],
                                        ground_truth_ids=known_gt_ids
                                    )
                                    for exp in g_explanations:
                                        render_product_card(exp)

    with tab_evaluation:
        st.subheader("Benchmark Metrics on `validation.jsonl`")
        render_evaluation_dashboard()

    with tab_docs:
        render_documentation()


def render_product_card(exp: dict):
    """Render a clean, informative product recommendation card."""
    with st.container():
        st.markdown('<div class="product-card">', unsafe_allow_html=True)
        c_img, c_info = st.columns([1, 2])

        with c_img:
            img_path = exp.get("image_path")
            if img_path and Path(img_path).exists():
                try:
                    p_img = Image.open(img_path).convert("RGB")
                    st.image(p_img, use_container_width=True)
                except Exception:
                    st.image("https://placehold.co/200x200?text=No+Image", use_container_width=True)
            else:
                st.image("https://placehold.co/200x200?text=Catalog+Image", use_container_width=True)

        with c_info:
            # Header with Rank and Match Badge (inline layout, no nested columns)
            badge_html = (
                '<span class="badge-exact">Exact Match</span>'
                if exp.get("is_exact_match")
                else '<span class="badge-similar">Visual Similarity</span>'
            )
            st.markdown(
                f"**#{exp['rank']} Product ID:** `{exp['product_id']}` &nbsp; {badge_html}",
                unsafe_allow_html=True
            )

            # Similarity Score Bar
            sim = exp["cosine_similarity"]
            st.progress(max(0.0, min(1.0, (sim + 1.0) / 2.0 if sim < 0 else sim)))
            st.caption(f"**Cosine Similarity:** `{sim:.4f}` *(normalized vector dot product, not a probability)*")

            # Extracted visual evidence
            st.markdown(
                f'<span class="tag">Color: {exp.get("dominant_color", "N/A")}</span>'
                f'<span class="tag">Pattern: {exp.get("pattern", "N/A")}</span>',
                unsafe_allow_html=True
            )

            st.write(f"**Evidence:** {exp.get('summary', '')}")

        st.markdown('</div>', unsafe_allow_html=True)


def render_evaluation_dashboard():
    """Render benchmark metrics from evaluation output files."""
    if not EVAL_RESULTS_JSON.exists():
        st.info("Evaluation results not found yet. Run `python scripts/evaluate.py` to generate real benchmark metrics.")
        return

    try:
        with open(EVAL_RESULTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        metrics = data.get("aggregate_metrics", {})
        per_scene = data.get("per_scene_evaluations", [])

        if not metrics:
            st.warning("Evaluation file is empty.")
            return

        # Top Metric Cards
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Recall @ 1", f"{metrics.get('recall@1', 0.0):.4f}")
        with m_col2:
            st.metric("Recall @ 5", f"{metrics.get('recall@5', 0.0):.4f}")
        with m_col3:
            st.metric("Recall @ 10", f"{metrics.get('recall@10', 0.0):.4f}")
        with m_col4:
            st.metric("Mean Reciprocal Rank (MRR)", f"{metrics.get('mrr', 0.0):.4f}")

        st.caption(f"Evaluated across **{metrics.get('num_evaluated_scenes', 0)}** validation scenes.")

        st.markdown("---")
        st.subheader("Per-Scene Retrieval Audit Table")

        if EVAL_RESULTS_CSV.exists():
            df = pd.read_csv(EVAL_RESULTS_CSV)
            st.dataframe(df, use_container_width=True)
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download CSV Report",
                data=csv_data,
                file_name="per_scene_evaluation.csv",
                mime="text/csv"
            )
        else:
            st.json(per_scene[:10])

    except Exception as e:
        st.error(f"Error reading evaluation results: {e}")


def render_documentation():
    """Render technical documentation and methodology."""
    st.markdown("""
    ### Technical Methodology

    #### 1. Representation & Embedding Model
    - **Backbone Architecture:** OpenCLIP ViT-B/32 (Vision Transformer with 32x32 patch size, 512-dimensional output embedding).
    - **Normalization:** Embeddings are L2-normalized so inner product equals cosine similarity:
      $$\\text{Cosine Similarity}(u, v) = \\frac{u \\cdot v}{\\|u\\| \\|v\\|} = u \\cdot v$$
    - **Inference Optimization:** Batched tensor preprocessing and evaluation across CPU/GPU with thread-safe resource caching.

    #### 2. Vector Indexing & Search
    - **Search Engine:** FAISS `IndexFlatIP` (Exact inner product search over normalized embeddings).
    - **Persistence:** Metadata and embeddings stored locally in `artifacts/embeddings/` to enable zero-overhead startups.

    #### 3. Garment-Level Localization
    - Combines spatial anatomical priors (Upper body, Lower body, Footwear) with OpenCLIP zero-shot classification to isolate garments from complex scenes.

    #### 4. Grounded Explanations
    - Visual explanations extract dominant color palettes and pattern textures directly from image pixels and CLIP representations, avoiding fabricated product descriptions.
    """)


if __name__ == "__main__":
    main()
