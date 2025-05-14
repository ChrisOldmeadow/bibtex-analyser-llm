"""Tag generation module for Bibtex Analyzer using OpenAI's GPT models."""

import logging
import random
import time
from typing import List, Dict, Any, Optional, Set
import openai
from openai import OpenAI

from .utils.text_processing import normalize_tags, collapse_tags, filter_redundant_tags

logger = logging.getLogger(__name__)

class TagGenerator:
    """Class for generating and managing tags using OpenAI's GPT models."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        """Initialize the TagGenerator with API credentials.
        
        Args:
            api_key: OpenAI API key. If None, will use OPENAI_API_KEY environment variable.
            model: Default model to use for tag generation.
        """
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.default_model = model
        self.rate_limit_delay = 1.0  # Delay between API calls in seconds
        
    def generate_tags_for_abstracts(
        self,
        abstracts: List[Dict[str, Any]],
        categories: Dict[str, str] = None,
        samples_per_category: int = 30,
        method_model: str = "gpt-4"
    ) -> Set[str]:
        """Generate tags for a list of abstracts by category.
        
        Args:
            abstracts: List of dictionaries containing 'abstract' and other metadata
            categories: Dictionary mapping category names to example tags
            samples_per_category: Number of abstracts to sample per category
            method_model: Model to use for the 'Statistical Methods' category
            
        Returns:
            Set of generated tags
        """
        if not categories:
            categories = {
                "General Topics": "e.g. cancer, mental health, equity, adolescents, health services",
                "Study Designs": "e.g. randomised controlled trial, stepped wedge, pragmatic trial, observational study",
                "Statistical Methods": "e.g. regression, survival analysis, Bayesian methods, causal inference, mixed models, machine learning",
                "Data Sources": "e.g. linked data, electronic health records, administrative data, surveys, registries",
                "Target Populations": "e.g. Indigenous Australians, school children, smokers, older adults, low-income groups, rural communities"
            }
        
        combined_tags = set()
        
        # Filter out entries without abstracts
        valid_abstracts = [a for a in abstracts if a.get('abstract')]
        if not valid_abstracts:
            logger.warning("No abstracts found in the provided entries. Please check if the 'abstract' field exists in your data.")
            return set()
            
        # If we don't have enough valid abstracts, adjust samples_per_category
        samples_per_category = min(samples_per_category, len(valid_abstracts))
        
        for category, examples in categories.items():
            logger.info(f"Generating tags for category: {category}")
            
            # Use a more powerful model for statistical methods
            model = method_model if category == "Statistical Methods" else self.default_model
            
            try:
                tags = self._get_tags_by_category(
                    valid_abstracts, 
                    category, 
                    examples, 
                    model=model, 
                    n=samples_per_category
                )
                logger.info(f"Generated {len(tags)} tags for {category}: {', '.join(tags)}")
                combined_tags.update(tags)
                
                # Respect rate limits
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Error generating tags for {category}: {str(e)}")
                continue
                
        return combined_tags
    
    def _get_tags_by_category(
        self, 
        abstracts: List[Dict[str, Any]], 
        category_name: str, 
        examples: str,
        model: str = "gpt-3.5-turbo",
        n: int = 30
    ) -> List[str]:
        """Generate tags for a specific category.
        
        Args:
            abstracts: List of abstract dictionaries
            category_name: Name of the category
            examples: Example tags for the category
            model: Model to use for generation
            n: Number of abstracts to sample
            
        Returns:
            List of generated tags
        """
        # Filter out entries without abstracts
        valid_abstracts = [a for a in abstracts if a.get('abstract')]
        if not valid_abstracts:
            logger.warning(f"No valid abstracts found for category: {category_name}")
            return []
            
        # Sample from valid abstracts
        batch = random.sample(valid_abstracts, min(n, len(valid_abstracts)))
        
        # Prepare abstract text, ensuring it's not too long
        abstract_texts = []
        for i, a in enumerate(batch):
            abstract = str(a.get('abstract', ''))
            abstract_texts.append(f"{i+1}. {abstract[:500]}")  # Limit abstract length
            
        text = "\n\n".join(abstract_texts)
        
        prompt = f"""You are an expert in research methods. Given the following {len(batch)} research abstracts:

{text}

Identify 5–10 topic tags related to **{category_name}**. Examples include: {examples}

Only return tags that clearly apply to multiple abstracts. Each tag should be a single word or a short phrase (2-3 words).

Output a comma-separated list of tags."""

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,  # Slightly higher temperature for more creative tags
                max_tokens=200
            )

            raw_tags = response.choices[0].message.content.strip().split(",")
            return normalize_tags(raw_tags, style="title")
            
        except Exception as e:
            logger.error(f"Error generating tags for category '{category_name}': {str(e)}")
            return []
    
    def assign_tags_to_abstracts(
        self, 
        entries: List[Dict[str, Any]], 
        tags: List[str],
        model: str = "gpt-3.5-turbo"
    ) -> List[Dict[str, Any]]:
        """Assign tags to each abstract from a predefined set.
        
        Args:
            entries: List of entry dictionaries
            tags: List of tags to choose from
            model: Model to use for tag assignment
            
        Returns:
            List of entries with assigned tags
        """
        if not tags:
            logger.warning("No tags provided for assignment. Returning original entries.")
            return entries
            
        tag_list = ", ".join(tags)
        
        for i, entry in enumerate(entries):
            # Skip entries without an abstract
            if not entry.get('abstract'):
                entry["tags"] = ""
                logger.warning(f"Skipping entry {i+1}: No abstract found")
                continue
                
            try:
                # Prepare the abstract text, ensuring it's not too long
                abstract = str(entry['abstract'])
                if len(abstract) > 3000:  # Limit abstract length
                    abstract = abstract[:3000] + "..."
                
                prompt = f"""Given the following predefined topic tags: {tag_list}

Assign 3–5 of the most relevant tags to this abstract. Only use tags from the list.

"{abstract}"

Tags:"""

                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,  # Lower temperature for more consistent results
                    max_tokens=100
                )
                
                raw_tags = response.choices[0].message.content.strip()
                entry["tags"] = ", ".join(normalize_tags(raw_tags.split(","), style="title"))
                
                # Log progress
                if (i + 1) % 5 == 0:  # Log more frequently
                    logger.info(f"Tagged {i + 1}/{len(entries)} abstracts")
                
                # Respect rate limits
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Error tagging abstract {i+1}: {str(e)}"
                             f"\nTitle: {entry.get('title', 'Unknown')}")
                entry["tags"] = ""
                continue
                
        return entries
    
    def interactive_tag_generation(
        self, 
        entries: List[Dict[str, Any]], 
        start_n: int = 30, 
        max_n: int = 60, 
        step: int = 10
    ) -> List[str]:
        """Interactively generate and confirm tags.
        
        Args:
            entries: List of entry dictionaries
            start_n: Initial number of abstracts to sample
            max_n: Maximum number of abstracts to sample
            step: Number of additional abstracts to sample if user is not satisfied
            
        Returns:
            List of confirmed tags
        """
        # Filter out entries without abstracts
        valid_entries = [e for e in entries if e.get('abstract')]
        if not valid_entries:
            logger.warning("No entries with abstracts found. Cannot generate tags.")
            return []
            
        # Adjust sample sizes based on available entries
        start_n = min(start_n, len(valid_entries))
        max_n = min(max_n, len(valid_entries))
        
        if start_n < 3:  # Need at least 3 abstracts for meaningful tags
            logger.warning(f"Only {len(valid_entries)} entries with abstracts found. Need at least 3 for meaningful tags.")
            return []
            
        print(f"🔍 Starting interactive tag generation...\n")
        print(f"📊 Sampling {start_n} abstracts...")
        
        # Initial tag generation
        tags = self.generate_tags_for_abstracts(
            valid_entries,
            samples_per_category=start_n
        )
        
        if not tags:
            print("⚠️ No tags were generated. The abstracts might be too short or not in English.")
            return []
            
        print(f"\n🎯 Generated {len(tags)} unique tags:")
        print(", ".join(sorted(tags)))
        
        # Ask user if they want to refine
        while True:
            response = input("\nAre you satisfied with these tags? (y/n, or 'more' to analyze more abstracts): ").strip().lower()
            
            if response == 'y':
                break
            elif response == 'more' and start_n < max_n:
                # Sample more abstracts
                start_n = min(start_n + step, max_n)
                print(f"\n📊 Analyzing {start_n} abstracts...")
                
                tags = self.generate_tags_for_abstracts(
                    valid_entries,
                    samples_per_category=start_n
                )
                
                if not tags:
                    print("⚠️ No tags were generated. The abstracts might be too short or not in English.")
                    return []
                    
                print(f"\n🎯 Generated {len(tags)} unique tags:")
                print(", ".join(sorted(tags)))
            else:
                if response != 'n':
                    print("Please enter 'y', 'n', or 'more'.")
                continue
                
        return sorted(tags)
