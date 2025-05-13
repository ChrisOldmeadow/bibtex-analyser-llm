import os
import sys
import random
from openai import OpenAI
import bibtexparser
from time import sleep
import pandas as pd

# Initialise OpenAI client
client = OpenAI()

# --- Load and clean BibTeX ---
def load_abstracts_from_bib(bib_path):
    with open(bib_path, 'r', encoding='utf-8') as bibfile:
        parser = bibtexparser.bparser.BibTexParser(common_strings=True)
        db = bibtexparser.load(bibfile, parser=parser)

    entries = []
    for entry in db.entries:
        title = entry.get("title", "").strip()
        abstract = entry.get("abstract", "").strip()
        if abstract:
            entries.append({
                "id": entry.get("ID", ""),
                "title": title,
                "abstract": abstract
            })
    return entries

# --- Tag Normalisation ---
def normalise_tags(tag_list, style="title"):
    cleaned = set()
    for tag in tag_list:
        tag = tag.strip()
        if style == "lower":
            tag = tag.lower()
        elif style == "title":
            tag = tag.title()
        cleaned.add(tag)
    return sorted(cleaned)

# --- Improved tag generation by category ---
def get_tags_by_category(abstracts, category_name, examples, model="gpt-3.5-turbo", n=30):
    batch = random.sample(abstracts, min(n, len(abstracts)))
    text = "\n\n".join([f"{i+1}. {a['abstract'][:300]}" for i, a in enumerate(batch)])

    prompt = f"""You are an expert in research methods. Given the following {len(batch)} research abstracts:

{text}

Identify 5–10 topic tags related to **{category_name}**. Examples include: {examples}

Only return tags that clearly apply to multiple abstracts.

Output a comma-separated list of tags."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_tags = response.choices[0].message.content.strip().split(",")
    return normalise_tags(raw_tags, style="title")

# --- Combined multi-category tag generation ---
def generate_tags_from_abstracts(entries, n=30, method_model="gpt-4"):
    print("🔍 Sampling abstracts and generating tags by category...")

    categories = {
    "General Topics": "e.g. cancer, mental health, equity, adolescents, health services",
    "Study Designs": "e.g. randomised controlled trial, stepped wedge, pragmatic trial, observational study",
    "Statistical Methods": "e.g. regression, survival analysis, Bayesian methods, causal inference, mixed models, machine learning",
    "Data Sources": "e.g. linked data, electronic health records, administrative data, surveys, registries",
    "Target Populations": "e.g. Indigenous Australians, school children, smokers, older adults, low-income groups, rural communities"
}


    combined_tags = set()
    for category, examples in categories.items():
        print(f"🧠 Generating tags for {category}...")
        model = method_model if category == "Statistical Methods" else "gpt-3.5-turbo"
        tags = get_tags_by_category(entries, category, examples, model=model, n=n)
        print(f"✅ {category} tags: {tags}")
        combined_tags.update(tags)

    return sorted(combined_tags)

# --- Tag each abstract individually using a fixed tagset ---
def tag_abstract(abstract, tags):
    tag_list = ", ".join(tags)
    prompt = f"""Given the following predefined topic tags: {tag_list}

Assign 3–5 of the most relevant tags to this abstract. Only use tags from the list.

\"{abstract}\"

Tags:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content.strip()
    return ", ".join(normalise_tags(raw.split(","), style="title"))

# --- Interactive tagset confirmation loop ---
def confirm_tagset_loop(entries, start_n=30, max_n=60, step=10):
    n = start_n
    while n <= max_n:
        tags = generate_tags_from_abstracts(entries, n=n)
        print("\n🧠 Combined tag set so far:")
        print(", ".join(tags))

        answer = input("\n✅ Are you happy with these tags? (y/n): ").strip().lower()
        if answer == "y":
            print("👍 Tagset confirmed.")
            return tags
        else:
            print("🔁 Adding more tags from a larger sample...")
            n += step

    print("⚠️ Reached maximum sample size. Using final cumulative tagset.")
    return tags

# --- Full pipeline ---
def tag_bibtex_corpus(bib_path):
    print("🔍 Loading BibTeX entries...")
    entries = load_abstracts_from_bib(bib_path)

    print("🧠 Generating topic tags interactively...")
    tags = confirm_tagset_loop(entries, start_n=30, max_n=60, step=10)
    print("✅ Final topic tags:", tags)

    print("🏷 Tagging abstracts...")
    for entry in entries:
        assigned_tags = tag_abstract(entry["abstract"], tags)
        entry["tags"] = assigned_tags
        print(f"• {entry['id']}: {assigned_tags}")
        sleep(1)  # avoid rate limits

    return entries

# --- Save to CSV ---
def save_tagged_entries(entries, out_path="tagged_abstracts.csv"):
    df = pd.DataFrame(entries)
    df.to_csv(out_path, index=False)
    print(f"✅ Saved to {out_path}")

# --- Entry point ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bibtex-gpt.py path/to/your_file.bib")
        sys.exit(1)

    bib_path = sys.argv[1]
    tagged_entries = tag_bibtex_corpus(bib_path)
    save_tagged_entries(tagged_entries)
