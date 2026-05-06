import streamlit as st
from pathlib import Path

# =========================
# EDIT THIS PATH
# =========================
root_dir = Path("./output/val_image_gen")



def get_example_dirs(root: Path):
    dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("example_")]
    dirs.sort()
    return dirs


def find_file(example_dir: Path, base_name: str):
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        p = example_dir / f"{base_name}{ext}"
        if p.exists():
            return p
    return None



def load_graph_text(example_dir: Path):
    p = example_dir / "graph.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return None



st.set_page_config(layout="wide")
st.title("Scene Graph → Image Viewer")

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

    graph_img = find_file(ex_dir, "rendered_graph")
    gen_img = find_file(ex_dir, "generated_image")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Graph**")
        if graph_img:
            st.image(str(graph_img), use_container_width=True)
        else:
            st.info("No graph image found")

    with col2:
        st.markdown("**Generated Image**")
        if gen_img:
            st.image(str(gen_img), use_container_width=True)
        else:
            st.info("No generated image found")

    if show_graph_text:
        graph_text = load_graph_text(ex_dir)
        if graph_text:
            st.text(graph_text)