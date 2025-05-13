import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import wordcloud  # for debug info

print("✔️ Using wordcloud from:", wordcloud.__file__)
print("✔️ Wordcloud version:", wordcloud.__version__)
import matplotlib.pyplot as plt

# Load tagged abstracts
df = pd.read_csv("tagged_abstracts.csv")

# Flatten all tags
all_tags = []
for tag_list in df['tags']:
    tags = [tag.strip().title() for tag in tag_list.split(",")]
    all_tags.extend(tags)

# Optional: collapse similar or redundant tags
def collapse_tags(tag_list):
    synonym_map = {
        "Rct": "Randomised Controlled Trial",
        "RCT": "Randomised Controlled Trial",
        "Intervention Study": "Randomised Controlled Trial",  # Assume RCT subsumes it
        "Pragmatic Trial": "Randomised Controlled Trial",
        "Mixed-Effects Models": "Mixed Models",
        "Children": "School Children",
        "Smoking": "Smokers"
    }
    collapsed = []
    for tag in tag_list:
        tag = synonym_map.get(tag, tag)
        collapsed.append(tag)
    return collapsed

# Collapse and count frequency
clean_tags = collapse_tags(all_tags)
tag_counts = Counter(clean_tags)

# Optional: remove overly broad if more specific tags exist
def filter_redundant(tags, drop_if_subset_of):
    filtered = Counter()
    for tag, count in tags.items():
        redundant = any(other != tag and tag in other for other in tags if other in drop_if_subset_of)
        if not redundant:
            filtered[tag] = count
    return filtered

# e.g., remove "Intervention Study" if "Randomised Controlled Trial" is present
redundant_filtered = filter_redundant(tag_counts, drop_if_subset_of={"Randomised Controlled Trial"})

# Keep only tags used more than once, then limit to top 50
tag_counts_filtered = {k: v for k, v in redundant_filtered.items() if v > 1}
tag_counts_limited = dict(Counter(tag_counts_filtered).most_common(50))
# Generate word cloud
wc = WordCloud(width=1200, height=800, background_color="white")
wc.generate_from_frequencies(tag_counts_limited)

# Display
plt.figure(figsize=(15, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.tight_layout()
plt.show()
