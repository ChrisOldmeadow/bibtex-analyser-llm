import pandas as pd
import random
from collections import Counter
import plotly.graph_objs as go
from pathlib import Path

# Load tagged abstracts
df = pd.read_csv("tagged_abstracts.csv")

# Build tag-to-title mapping
tag_to_titles = {}
all_tags = []

for _, row in df.iterrows():
    title = row["title"]
    tags = [tag.strip().title() for tag in row["tags"].split(",")]
    for tag in tags:
        all_tags.append(tag)
        tag_to_titles.setdefault(tag, []).append(title)

# Limit to top N tags
top_n = 50
tag_counts = Counter(all_tags)
top_tags = dict(tag_counts.most_common(top_n))

# Make a safe URL-friendly version of a tag
def slugify(tag):
    return tag.lower().replace(" ", "-")

# Hover text (up to 10 titles)
def make_hover_text(tag):
    titles = tag_to_titles[tag]
    return f"<b>{tag}</b><br>Count: {len(titles)}<br><br>" + "<br>".join(titles[:10]) + (
        "<br>..." if len(titles) > 10 else "")

# Make link for each tag (assuming pages live in data/tags/)
def make_link(tag):
    return f"/data/tags/{tag.lower().replace(' ', '-')}.html"
# Layout for word positions
x = [random.random() for _ in top_tags]
y = [random.random() for _ in top_tags]

# Create plot
fig = go.Figure()

for (tag, freq), x_pos, y_pos in zip(top_tags.items(), x, y):
    fig.add_trace(go.Scatter(
        x=[x_pos],
        y=[y_pos],
        mode="text",
        text=[f"<a href='{make_link(tag)}'>{tag}</a>"],
        textfont=dict(size=10 + freq * 2),
        hovertext=make_hover_text(tag),
        hoverinfo="text",
        textposition="middle center",
        hoverlabel=dict(bgcolor="white")
    ))

fig.update_layout(
    title="Interactive Word Cloud",
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    showlegend=False,
    margin=dict(l=20, r=20, t=40, b=20),
)

# Output path
output_file = Path("interactive_wordcloud.html")
fig.write_html(output_file, include_plotlyjs="cdn")

print(f"✅ Saved: {output_file.resolve()}")
