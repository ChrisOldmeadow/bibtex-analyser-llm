import pandas as pd
from collections import Counter
import json
from pathlib import Path

# --- CONFIG ---
CSV_FILE = "tagged_abstracts.csv"
OUTPUT_DIR = Path("data/wordcloud2")
TOP_N = 100  # Limit to top N tags

# --- LOAD TAGS ---
df = pd.read_csv(CSV_FILE)

all_tags = []
for tag_list in df["tags"]:
    tags = [tag.strip().title() for tag in tag_list.split(",")]
    all_tags.extend(tags)

tag_counts = Counter(all_tags)
top_tags = tag_counts.most_common(TOP_N)
words_data = [[tag, freq] for tag, freq in top_tags]

# --- CREATE OUTPUT FOLDER ---
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- WRITE JS DATA FILE ---
word_data_path = OUTPUT_DIR / "word_data.js"
with open(word_data_path, "w", encoding="utf-8") as f:
    f.write("const wordList = ")
    json.dump(words_data, f, indent=2)
    f.write(";")

# --- WRITE HTML FILE USING wordcloud2.js ---
html_path = OUTPUT_DIR / "wordcloud.html"
with open(html_path, "w", encoding="utf-8") as f:
    f.write("""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Interactive Word Cloud</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/wordcloud2.js/1.1.2/wordcloud2.min.js"></script>
  <script src="/data/wordcloud2/word_data.js"></script>
  <style>
    body { font-family: sans-serif; padding: 2rem; }
    #wordcloud {
      width: 100%;
      height: 600px;
      border: 1px solid #ccc;
    }
  </style>
</head>
<body>
  <h2>Interactive Word Cloud</h2>
  <div id="wordcloud"></div>
  <script>
    WordCloud(document.getElementById('wordcloud'), {
      list: wordList,
      gridSize: 10,
      weightFactor: function (size) {
        return size * 2.5;
      },
      fontFamily: 'Impact',
      rotateRatio: 0.4,
      rotationSteps: 2,
      backgroundColor: '#fff',
      click: function(item) {
        const tagSlug = item[0].toLowerCase().replace(/ /g, '-');
        window.open('/data/tags/' + tagSlug + '.html', '_blank');
      }
    });
  </script>
</body>
</html>
""")

print("✅ Word cloud exported to:", html_path.resolve())
