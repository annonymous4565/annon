import streamlit as st
from pathlib import Path
from PIL import Image

# -----------------------------
# Hardcoded folders
# -----------------------------

DATASET = "COCO"

pp =[
    "five_category",
    "mix_relation",
    "only_numeral",
    "sampled_non_relation",
    "sampled_only_semantic",
    "sampled_only_spatial",

]
ii = 0

# FOLDERS = {
#     "Folder 1": Path(f"./output/ImageGen/SDXL/{DATASET}/{pp[ii]}/cfgpp0.8_/samples"),
#     "Folder 2": Path(f"./output/ImageGen/ComposeDiff/{DATASET}/{pp[ii]}/cfg5.0_/samples"),
#     "Folder 3": Path(f"./output/ImageGen/CO3/{DATASET}/{pp[ii]}/cfgpp0.8_LatentCorr_step5_modCompWts_beta0.9_algouflow_same_t_sum1_uncondcorr_hybrid_6_nresmpl_4_/samples"),
#     "Folder 4": Path(f"./output/ImageGen/DiscreteSG/{DATASET}/{pp[ii]}/samples"),
#     "Folder 5": Path("/path/to/folder_5"),
# }

# FOLDERS = {
#     "Folder 1": Path(f"./output/ImageGen/SDXL/{DATASET}/cfgpp0.8_/samples"),
#     "Folder 2": Path(f"./output/ImageGen/ComposeDiff/{DATASET}/cfg5.0_/samples"),
#     "Folder 3": Path(f"./output/ImageGen/CO3/{DATASET}/cfgpp0.8_LatentCorr_step5_modCompWts_beta0.9_algouflow_same_t_sum1_uncondcorr_hybrid_6_nresmpl_4_/samples"),
#     "Folder 4": Path(f"./output/ImageGen/DiscreteSG/{DATASET}/samples"),
#     "Folder 5": Path("/path/to/folder_5"),
# }

FOLDERS = {
    "Folder 1": Path(f"./output/MAIN_RES/{DATASET}/LDM"),
    "Folder 2": Path(f"./output/MAIN_RES/{DATASET}/DiscreteSG/")
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

st.set_page_config(page_title="5-Folder Image Comparison", layout="wide")
st.title("5-Folder Image Comparison")

# -----------------------------
# Collect images by filename stem
# -----------------------------
images_by_id = {}

for folder_name, folder_path in FOLDERS.items():
    if not folder_path.exists():
        st.warning(f"Missing folder: {folder_path}")
        continue

    for img_path in folder_path.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        img_id = img_path.stem  # filename without extension

        if img_id not in images_by_id:
            images_by_id[img_id] = {}

        images_by_id[img_id][folder_name] = img_path

all_ids = sorted(images_by_id.keys())

st.sidebar.header("Filter IDs")

selected_ids = st.sidebar.multiselect(
    "Select IDs",
    options=all_ids,
    default=all_ids,
)

manual_ids_text = st.sidebar.text_area(
    "Or enter IDs manually, one per line",
    placeholder="000001\n000002\n000003",
)

if manual_ids_text.strip():
    manual_ids = [x.strip() for x in manual_ids_text.splitlines() if x.strip()]
    display_ids = [x for x in manual_ids if x in images_by_id]
else:
    display_ids = selected_ids

st.write(f"Showing {len(display_ids)} / {len(all_ids)} IDs")

# -----------------------------
# Display comparison rows
# -----------------------------
for img_id in display_ids:
    st.markdown("---")
    st.subheader(img_id)

    cols = st.columns(len(FOLDERS))

    for col, (folder_name, _) in zip(cols, FOLDERS.items()):
        with col:
            st.markdown(f"**{folder_name}**")

            img_path = images_by_id[img_id].get(folder_name)

            if img_path is None:
                st.info("Missing")
            else:
                st.image(Image.open(img_path), use_container_width=True)
                st.caption(img_path.name)