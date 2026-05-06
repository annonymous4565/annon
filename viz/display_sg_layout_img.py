import streamlit as st
from pathlib import Path

# =========================
# EDIT THIS PATH
# =========================
root_dir = Path("./output/layout_image_gen")


def get_example_dirs(root: Path):
    dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("example_")]
    dirs.sort()
    return dirs


def find_file(example_dir: Path, names):
    for name in names:
        p = example_dir / name
        if p.exists():
            return p
    return None


def load_text(example_dir: Path, filename: str):
    p = example_dir / filename
    if p.exists():
        return p.read_text(encoding="utf-8")
    return None


st.set_page_config(layout="wide")
st.title("Layout-conditioned SG → Image Viewer")

if not root_dir.exists():
    st.error(f"Directory not found: {root_dir}")
    st.stop()

example_dirs = get_example_dirs(root_dir)
example_names = [p.name for p in example_dirs]

st.write(f"Found {len(example_dirs)} examples")

show_graph_text = st.checkbox("Show graph text", value=False)

selected_names = st.multiselect(
    "Select examples to display",
    options=example_names,
    default=example_names[: min(10, len(example_names))],
)

if len(selected_names) == 0:
    st.info("No examples selected.")
    st.stop()

selected_dirs = [p for p in example_dirs if p.name in selected_names]

for ex_dir in selected_dirs:
    st.markdown("---")
    st.subheader(ex_dir.name)

    graph_img = find_file(ex_dir, [
        "rendered_graph.png", "rendered_graph.jpg", "rendered_graph.jpeg", "rendered_graph.webp"
    ])
    layout_boxes_img = find_file(ex_dir, [
        "layout_boxes.png", "layout_boxes.jpg", "layout_boxes.jpeg", "layout_boxes.webp"
    ])
    gen_img = find_file(ex_dir, [
        "generated_image.jpg", "generated_image.png", "generated_image.jpeg", "generated_image.webp"
    ])
    gen_img_boxes = find_file(ex_dir, [
        "generated_image_with_boxes.jpg",
        "generated_image_with_boxes.png",
        "generated_image_with_boxes.jpeg",
        "generated_image_with_boxes.webp",
    ])

    cols = st.columns(4)

    with cols[0]:
        st.markdown("**Graph**")
        if graph_img:
            st.image(str(graph_img), use_container_width=True)
        else:
            st.info("No graph image")

    with cols[1]:
        st.markdown("**Layout Boxes**")
        if layout_boxes_img:
            st.image(str(layout_boxes_img), use_container_width=True)
        else:
            st.info("No layout boxes image")

    with cols[2]:
        st.markdown("**Generated Image**")
        if gen_img:
            st.image(str(gen_img), use_container_width=True)
        else:
            st.info("No generated image")

    with cols[3]:
        st.markdown("**Generated + Boxes**")
        if gen_img_boxes:
            st.image(str(gen_img_boxes), use_container_width=True)
        else:
            st.info("No overlay image")

    if show_graph_text:
        graph_text = load_text(ex_dir, "sampled_graph.txt")
        if graph_text is None:
            graph_text = load_text(ex_dir, "graph.txt")
        if graph_text:
            st.text(graph_text)