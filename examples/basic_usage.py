"""
Basic usage example for the Bibtex Analyzer package.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    """Demonstrate basic usage of the Bibtex Analyzer."""
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key.")
        return

    # Import here to show the dependency clearly
    from bibtex_analyzer import process_bibtex_file, TagGenerator
    from bibtex_analyzer.visualization import (
        create_mpl_wordcloud,
        create_plotly_wordcloud,
        plot_tag_frequencies
    )

    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    print("Step 1: Processing BibTeX file...")
    input_file = Path("examples/sample_references.bib")
    
    # Process the BibTeX file
    entries = process_bibtex_file(
        input_file,
        output_file=output_dir / "tagged_papers.csv"
    )
    
    print(f"Processed {len(entries)} entries.")
    
    # Initialize the tag generator
    print("\nStep 2: Generating tags...")
    tag_generator = TagGenerator()
    
    # Generate tags (using a small sample for demonstration)
    sample_size = min(5, len(entries))
    sample_entries = entries[:sample_size]
    
    print(f"Generating tags for {sample_size} sample entries...")
    tags = tag_generator.generate_tags_for_abstracts(
        sample_entries,
        samples_per_category=2
    )
    
    print(f"Generated {len(tags)} unique tags.")
    
    # Assign tags to all entries
    print("\nStep 3: Assigning tags to all entries...")
    tagged_entries = tag_generator.assign_tags_to_abstracts(entries, list(tags))
    
    # Save the tagged entries
    import pandas as pd
    df = pd.DataFrame(tagged_entries)
    output_file = output_dir / "tagged_papers_final.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved tagged papers to {output_file}")
    
    # Create visualizations
    print("\nStep 4: Creating visualizations...")
    
    # Create a static word cloud
    print("Creating static word cloud...")
    create_mpl_wordcloud(
        df,
        output_file=output_dir / "wordcloud.png",
        show=False
    )
    
    # Create an interactive word cloud
    print("Creating interactive word cloud...")
    create_plotly_wordcloud(
        df,
        output_file=output_dir / "wordcloud_interactive.html"
    )
    
    # Create tag frequency plot
    print("Creating tag frequency plot...")
    plot_tag_frequencies(
        df,
        output_file=output_dir / "tag_frequencies.png",
        show=False
    )
    
    print("\nAll done! Check the 'output' directory for results.")

if __name__ == "__main__":
    main()
