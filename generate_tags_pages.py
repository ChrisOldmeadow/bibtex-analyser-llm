import pandas as pd
from pathlib import Path

# --- Configuration ---
input_csv = "tagged_abstracts.csv"
output_dir = Path("data/tags")
max_titles = 100  # truncate long lists for legibility

# --- Helper function: slugify tag names for URLs ---
def slugify(tag):
    return tag.lower().replace(" ", "-")

# --- Load data ---
df = pd.read_csv(input_csv)

# --- Group titles by tag ---
tag_to_titles = {}

for _, row in df.iterrows():
    title = row["title"]
    tags = [tag.strip().title() for tag in row["tags"].split(",")]
    for tag in tags:
        tag_to_titles.setdefault(tag, []).append(title)

# --- Create output directory ---
output_dir.mkdir(parents=True, exist_ok=True)

# --- Write an HTML file for each tag ---
for tag, titles in tag_to_titles.items():
    safe_filename = slugify(tag) + ".html"
    filepath = output_dir / safe_filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"<html><head><title>{tag}</title></head><body>\n")
        f.write(f"<h2>Papers tagged with: {tag}</h2>\n")
        f.write("<ul>\n")
        for title in titles[:max_titles]:
            f.write(f"<li>{title}</li>\n")
        if len(titles) > max_titles:
            f.write("<li>... (truncated)</li>\n")
        f.write("</ul>\n</body></html>\n")

print(f"✅ Generated {len(tag_to_titles)} tag pages in {output_dir}")
