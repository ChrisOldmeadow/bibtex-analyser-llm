import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Sample tags
sample_tags = [
    "Bayesian Methods", "Causal Inference", "Randomised Controlled Trial",
    "School Children", "Linked Data", "Bayesian Methods", "Causal Inference"
]

# Count frequencies
tag_counts = Counter(sample_tags)

# Generate word cloud
wc = WordCloud(width=800, height=600, background_color="white")
wc.generate_from_frequencies(tag_counts)

# Show plot
plt.figure(figsize=(10, 6))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.tight_layout()
plt.show()
