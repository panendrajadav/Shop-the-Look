import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict
from PIL import Image, ImageOps
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    MODEL_NAME, PRETRAINED, DEVICE, DEFAULT_TOP_K,
    PRODUCTS_INDEX_FILE, PRODUCTS_METADATA_FILE, PRODUCTS_EMBEDDINGS_FILE,
    VALIDATION_FILE, IMAGES_DIR, EVAL_RESULTS_JSON
)
from src.image_encoder import CLIPImageEncoder
from src.catalog_index import ProductIndex
from src.image_downloader import ImageDownloader
from src.garment_detector import GarmentDetector
from src.retrieval import SceneRetrieval
from src.explanations import RecommendationExplainer
from src.data_loader import ValidationLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Shop the Look | Visual Product Discovery",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .product-card {
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        padding: 14px;
        background: #ffffff;
        margin-bottom: 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .badge-exact {
        background-color: #10b981;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-similar {
        background-color: #3b82f6;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .tag {
        background-color: #f1f5f9;
        color: #334155;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 6px;
        display: inline-block;
        margin-top: 4px;
        margin-bottom: 4px;
    }
    .crop-badge {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 8px;
        text-align: center;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing OpenCLIP and FAISS Catalog Index...")
def load_system():
    """Load and cache CLIP encoder, FAISS index, retrieval, and explanation engines."""
    try:
        index = ProductIndex(PRODUCTS_EMBEDDINGS_FILE, PRODUCTS_METADATA_FILE, PRODUCTS_INDEX_FILE)
        if not index.load():
            return None, None, None, None, None, None

        encoder = CLIPImageEncoder(model_name=MODEL_NAME, pretrained=PRETRAINED, device=DEVICE)
        downloader = ImageDownloader(IMAGES_DIR)
        garment_detector = GarmentDetector(clip_encoder=encoder, use_torchvision_detector=True)
        retrieval = SceneRetrieval(encoder=encoder, catalog_index=index, garment_detector=garment_detector)
        explainer = RecommendationExplainer(catalog_index=index, clip_encoder=encoder)

        return index, encoder, downloader, retrieval, explainer, garment_detector
    except Exception as e:
        logger.error(f"Error loading system: {e}")
        return None, None, None, None, None, None


@st.cache_data
def load_sample_scenes():
    """Load sample scene IDs from validation dataset."""
    try:
        val_loader = ValidationLoader(VALIDATION_FILE)
        val_loader.load()
        scenes = sorted(val_loader.get_all_scenes())
        return scenes, val_loader.scene_to_products
    except Exception:
        return [], {}


def render_product_card(exp: dict):
    """Render a clean, uncluttered catalog product recommendation card."""
    with st.container():
        st.markdown('<div class="product-card">', unsafe_allow_html=True)
        col_img, col_info = st.columns([1, 2], gap="medium")

        with col_img:
            img_path = exp.get("image_path")
            if img_path and Path(img_path).exists():
                try:
                    p_img = Image.open(img_path).convert("RGB")
                    st.image(p_img, use_container_width=True)
                except Exception:
                    st.caption("Image unavailable")
            else:
                st.caption("Image unavailable")

        with col_info:
            is_exact = exp.get("is_exact_match", False)
            badge_html = (
                '<span class="badge-exact">Exact Match</span>'
                if is_exact
                else '<span class="badge-similar">Visually Similar</span>'
            )
            st.markdown(
                f"**#{exp['rank']} Product ID:** `{exp['product_id']}` &nbsp; {badge_html}",
                unsafe_allow_html=True
            )

            sim = exp.get("cosine_similarity", 0.0)
            st.write(f"**Visual Similarity:** `{sim:.4f}`")

            cat_label = exp.get("category", "Apparel")
            color_label = exp.get("dominant_color", "N/A")
            pat_label = exp.get("pattern", "N/A")

            st.markdown(
                f'<span class="tag">Category: {cat_label}</span>'
                f'<span class="tag">Color: {color_label}</span>'
                f'<span class="tag">Pattern: {pat_label}</span>'
                f'<span class="tag">Style: {exp.get("style", "N/A")}</span>'
                f'<span class="tag">Material: {exp.get("material", "N/A")}</span>',
                unsafe_allow_html=True
            )

            summary_text = exp.get("summary", "")
            if summary_text:
                st.write(f"*{summary_text}*")
            if exp.get("shared"):
                st.write("**Matches the query on:** " + "; ".join(exp["shared"]))
            if exp.get("differs"):
                st.write("**Differs on:** " + "; ".join(exp["differs"]))

        st.markdown('</div>', unsafe_allow_html=True)


def main():
    st.title("🛍️ Shop the Look")
    st.caption("Detect fashion items in real-world outfit photos and discover matching products from the catalog.")

    index, encoder, downloader, retrieval, explainer, garment_detector = load_system()

    if index is None:
        st.error("⚠️ Catalog index not found. Please run `python scripts/build_index.py` first.")
        return

    sample_scenes, scene_to_gt = load_sample_scenes()

    with st.sidebar:
        st.header("⚙️ Settings")
        top_k = st.slider("Recommendations per garment", min_value=1, max_value=5, value=3)

        st.markdown("---")
        st.header("📊 Catalog Status")
        st.write(f"**Products Indexed:** {index.get_size():,}")
        st.write(f"**Embedding Model:** OpenCLIP ({MODEL_NAME})")
        st.write(f"**Hardware Device:** `{DEVICE.upper()}`")
        if st.button("🔄 Reload Cache & Index"):
            st.cache_resource.clear()
            st.rerun()

    col_input, col_display = st.columns([1, 1.2], gap="large")

    with col_input:
        st.subheader("1. Input Outfit Image")
        source = st.radio(
            "Choose Image Source:",
            ["Upload Custom Image", "Pick from Validation Scenes"],
            horizontal=True
        )

        query_image: Optional[Image.Image] = None
        current_scene_id = None
        known_gt_ids = set()

        if source == "Upload Custom Image":
            uploaded_file = st.file_uploader(
                "Upload an outfit or fashion photo (JPG, PNG)",
                type=["jpg", "jpeg", "png"]
            )
            if uploaded_file:
                try:
                    raw_img = Image.open(uploaded_file).convert("RGB")
                    query_image = ImageOps.exif_transpose(raw_img) or raw_img
                except Exception as e:
                    st.error(f"Failed to read image file: {e}")

        else:
            if sample_scenes:
                selected_idx = st.selectbox(
                    "Select a benchmark scene:",
                    options=range(len(sample_scenes)),
                    format_func=lambda i: f"Scene #{i+1} ({sample_scenes[i][:8]}...)"
                )
                current_scene_id = sample_scenes[selected_idx]
                known_gt_ids = set(scene_to_gt.get(current_scene_id, []))

                cached_path = downloader.get_cached_image(current_scene_id)
                if cached_path and cached_path.exists():
                    try:
                        raw_img = Image.open(cached_path).convert("RGB")
                        query_image = ImageOps.exif_transpose(raw_img) or raw_img
                    except Exception as e:
                        st.error(f"Error opening cached image: {e}")
                else:
                    with st.spinner("Downloading scene image from CDN..."):
                        dl_path = downloader.download_image(current_scene_id)
                        if dl_path and dl_path.exists():
                            raw_img = Image.open(dl_path).convert("RGB")
                            query_image = ImageOps.exif_transpose(raw_img) or raw_img
                        else:
                            st.warning("Could not download scene image. Please try another scene.")
            else:
                st.info("Validation scenes not found in `data/validation.jsonl`.")

    detected_garments = []
    if query_image is not None:
        with st.spinner("Detecting garments and finding matching products..."):
            detected_garments = retrieval.retrieve_garments(
                scene_image=query_image,
                top_k_per_garment=top_k,
                remove_duplicates=True
            )

        with col_input:
            if detected_garments:
                annotated_img = garment_detector.draw_detections(query_image, detected_garments)
                st.image(annotated_img, caption="Detected Garments with Bounding Boxes", use_container_width=True)
            else:
                st.image(query_image, caption="Query Outfit Image", use_container_width=True)

            if current_scene_id:
                st.caption(f"Scene ID: `{current_scene_id}` | Benchmark Positives: {len(known_gt_ids)}")

    with col_display:
        st.subheader("2. Recommended Catalog Products")

        if query_image is None:
            st.info("👈 Upload an outfit photo or select a benchmark scene to start.")
        elif not detected_garments:
            st.warning("No individual garments could be isolated from this image.")
        else:
            parts = [g for g in detected_garments if g["garment_label"] != "Full Outfit / Look"] or detected_garments
            labels = ["✨ Full Outfit"] + [
                f"{g.get('icon', '🏷️')} {g.get('item_name', g['garment_label'])} ({g.get('dominant_color', '')})"
                for g in parts
            ]
            selected_idx = st.radio(
                "Select the full outfit or one detected item:",
                options=list(range(len(labels))),
                format_func=lambda i: labels[i],
                horizontal=True
            )

            if selected_idx == 0:
                st.markdown("#### Best catalog match for each detected item")
                for g in parts:
                    if not g.get("recommendations"):
                        continue
                    best = explainer.explain_batch(
                        results=g["recommendations"],
                        query_image=g["crop_image"],
                        query_category=g["garment_label"],
                    )[0]
                    kind = "exact match" if best["is_exact_match"] else "most similar product"
                    st.markdown(
                        f"**{g.get('icon', '')} {g.get('item_name', g['garment_label'])}** "
                        f"({g.get('dominant_color', 'N/A')}): {kind}"
                    )
                    st.image(g["crop_image"], caption="Detected in photo", width=110)
                    render_product_card(best)
                selected_garment = None
            else:
                selected_garment = parts[selected_idx - 1]

        if query_image is not None and detected_garments and selected_garment is not None:

            g_col1, g_col2 = st.columns([1, 2], gap="medium")
            with g_col1:
                st.image(
                    selected_garment["crop_image"],
                    caption=f"{selected_garment['garment_label']}",
                    use_container_width=True
                )
            with g_col2:
                st.write(f"**Detected item:** {selected_garment.get('icon', '')} {selected_garment.get('item_name', selected_garment['garment_label'])}")
                st.write(f"**Category:** {selected_garment['garment_label']}")
                st.write(f"**Dominant Color:** `{selected_garment.get('dominant_color', 'N/A')}`")
                pat = selected_garment.get("pattern")
                if pat:
                    st.write(f"**Pattern:** `{pat}`")
                if selected_garment.get("is_uncertain"):
                    st.caption("⚠️ *Category classification uncertain; broader candidates retrieved.*")

            st.markdown("---")
            st.markdown("#### Matching Catalog Products")

            recs = selected_garment.get("recommendations", [])
            if not recs:
                st.warning("No matching products found in the catalog.")
            else:
                explanations = explainer.explain_batch(
                    results=recs,
                    query_image=selected_garment["crop_image"],
                    query_category=selected_garment["garment_label"],
                )
                if not any(e["is_exact_match"] for e in explanations):
                    st.info("No exact catalog match found for this garment. Showing the most visually similar products.")
                for exp in explanations:
                    render_product_card(exp)

    if current_scene_id and query_image is not None:
        st.markdown("---")
        with st.expander("🎯 Benchmark answer for this scene (validation label, not used for predictions)", expanded=False):
            indexed = set(index.product_ids)
            shown = {r[0] for g in detected_garments for r in g.get("recommendations", [])}
            gt_cols = st.columns(max(len(known_gt_ids), 3))
            for g_col, gt_id in zip(gt_cols, sorted(known_gt_ids)):
                with g_col:
                    gt_path = downloader.get_cached_image(gt_id)
                    if gt_path:
                        st.image(Image.open(gt_path).convert("RGB"), width=180)
                    st.caption(f"`{gt_id}`")
                    if gt_id not in indexed:
                        st.write("Not in the indexed catalog: only similar products can be returned.")
                    elif gt_id in shown:
                        st.write("✅ Retrieved in the results above.")
                    else:
                        st.write("❌ In the catalog but not in the top results.")

    if query_image is not None and detected_garments:
        st.markdown("---")
        with st.expander("🛠️ Developer Diagnostics (Raw Boxes, Confidences & Rerank Breakdown)", expanded=False):
            st.subheader("Detected Garment Crops & Bounding Boxes")
            crop_cols = st.columns(min(len(detected_garments), 5))
            for c_col, g in zip(crop_cols, detected_garments):
                with c_col:
                    st.image(g["crop_image"], caption=f"{g['garment_label']}\n{g.get('dominant_color', '')}", use_container_width=True)

            diag_table = []
            for idx, g in enumerate(detected_garments, 1):
                diag_table.append({
                    "Item #": idx,
                    "Category": g["garment_label"],
                    "Region": g["region"],
                    "Bounding Box [x1, y1, x2, y2]": str(g["box"]),
                    "Detector Confidence": f"{g['confidence']:.2%}",
                    "Dominant Color": g.get("dominant_color", "N/A"),
                    "Pattern": g.get("pattern", "N/A"),
                    "Status": "Uncertain" if g.get("is_uncertain") else "Reliable",
                })
            st.table(diag_table)

            st.subheader("Candidate Reranking Breakdown")
            for g in detected_garments:
                st.markdown(f"**Garment:** {g.get('icon', '')} `{g['garment_label']}` (Color: `{g.get('dominant_color')}`)")
                g_diags = g.get("diagnostics", [])
                if g_diags:
                    st.dataframe(g_diags, use_container_width=True)

    if EVAL_RESULTS_JSON.exists():
        with st.expander("📊 Benchmark Metrics (`validation.jsonl`)", expanded=False):
            try:
                with open(EVAL_RESULTS_JSON, "r", encoding="utf-8") as f:
                    data = json.load(f)
                metrics = data.get("aggregate_metrics", {})
                if metrics:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Recall @ 1", f"{metrics.get('recall@1', 0.0):.4f}")
                    m2.metric("Recall @ 5", f"{metrics.get('recall@5', 0.0):.4f}")
                    m3.metric("Recall @ 10", f"{metrics.get('recall@10', 0.0):.4f}")
                    m4.metric("MRR", f"{metrics.get('mrr', 0.0):.4f}")
                    st.caption(
                        f"Evaluated on {metrics.get('num_evaluated_scenes', 0)} scenes from `validation.jsonl` "
                        f"against {index.get_size():,} indexed catalog products using OpenCLIP {MODEL_NAME}."
                    )
            except Exception as e:
                st.caption(f"Could not load evaluation metrics: {e}")


if __name__ == "__main__":
    main()
