import streamlit as st
from pathlib import Path
from PIL import Image

# -----------------------------
# Hardcoded image folder
# -----------------------------

pp =[
    "five_category",
    "mix_relation",
    "only_numeral",
    "sampled_non_relation",
    "sampled_only_semantic",
    "sampled_only_spatial",

]
ii = 0
# ./output/SGgraph_baselines/completion/masked/DiscreteSG/Visual_Genome

IMAGE_DIR = Path(f"./output/SGgraph_baselines/completion/orig/DiscreteSG/COCO/{pp[ii]}")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

st.set_page_config(page_title="Image Folder Viewer", layout="wide")
st.title("Image Folder Viewer")

if not IMAGE_DIR.exists():
    st.error(f"Folder does not exist: {IMAGE_DIR}")
    st.stop()

image_paths = sorted(
    [p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS]
)

st.write(f"Found {len(image_paths)} images.")

cols_per_row = st.slider("Images per row", 1, 8, 4)

for i in range(0, len(image_paths), cols_per_row):
    cols = st.columns(cols_per_row)

    for col, img_path in zip(cols, image_paths[i:i + cols_per_row]):
        with col:
            st.caption(img_path.name)
            st.image(Image.open(img_path), use_container_width=True)