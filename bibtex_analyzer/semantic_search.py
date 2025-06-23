"""Semantic search functionality for finding topically similar papers."""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import pickle

import pandas as pd
import numpy as np
from openai import OpenAI
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class SemanticSearcher:
    """Semantic search for bibliographic data using OpenAI embeddings."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        """Initialize the semantic searcher.
        
        Args:
            api_key: OpenAI API key (will use environment variable if not provided)
            model: OpenAI embedding model to use
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.cache_dir = Path(".embeddings_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for the given text."""
        return hashlib.md5(f"{self.model}:{text}".encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embedding from cache if it exists."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: np.ndarray) -> None:
        """Save embedding to cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to save cache file {cache_file}: {e}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text, with caching.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return np.zeros(1536)  # Default embedding size for text-embedding-3-small
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        cached_embedding = self._load_from_cache(cache_key)
        if cached_embedding is not None:
            return cached_embedding
        
        try:
            # Get embedding from OpenAI
            response = self.client.embeddings.create(
                input=[text],
                model=self.model
            )
            embedding = np.array(response.data[0].embedding)
            
            # Cache the result
            self._save_to_cache(cache_key, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to get embedding for text: {e}")
            # Return zero vector as fallback
            return np.zeros(1536)
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 100, logger=None) -> np.ndarray:
        """Get embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            logger: Optional logger for progress tracking
            
        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
        """
        embeddings = []
        total_cached = 0
        total_new = 0
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            # Check cache for each text in batch
            uncached_texts = []
            uncached_indices = []
            
            for j, text in enumerate(batch):
                cache_key = self._get_cache_key(text)
                cached_embedding = self._load_from_cache(cache_key)
                
                if cached_embedding is not None:
                    batch_embeddings.append(cached_embedding)
                    total_cached += 1
                else:
                    batch_embeddings.append(None)
                    uncached_texts.append(text)
                    uncached_indices.append(j)
            
            # Log cache stats
            if logger and uncached_texts:
                cached_in_batch = len(batch) - len(uncached_texts)
                logger.log_info(f"Batch {i//batch_size + 1}: {cached_in_batch} cached, {len(uncached_texts)} need computing")
            
            # Get embeddings for uncached texts
            if uncached_texts:
                try:
                    if logger:
                        logger.log_info(f"Calling OpenAI API for {len(uncached_texts)} embeddings...")
                    
                    response = self.client.embeddings.create(
                        input=uncached_texts,
                        model=self.model
                    )
                    
                    # Insert new embeddings and cache them
                    for k, embedding_data in enumerate(response.data):
                        embedding = np.array(embedding_data.embedding)
                        batch_idx = uncached_indices[k]
                        batch_embeddings[batch_idx] = embedding
                        
                        # Cache the result
                        cache_key = self._get_cache_key(uncached_texts[k])
                        self._save_to_cache(cache_key, embedding)
                        total_new += 1
                        
                except Exception as e:
                    if logger:
                        logger.log_error(f"Failed to get embeddings for batch: {e}")
                    # Fill with zero vectors for failed embeddings
                    embedding_dim = 1536  # Default size
                    for k in uncached_indices:
                        if batch_embeddings[k] is None:
                            batch_embeddings[k] = np.zeros(embedding_dim)
            
            embeddings.extend(batch_embeddings)
            
            # Progress update
            if logger:
                progress = (i + len(batch)) / len(texts) * 100
                logger.log_info(f"Embedding progress: {int(progress)}% complete")
                # Update progress if logger has set_progress method
                if hasattr(logger, 'set_progress'):
                    # Map embedding progress to overall search progress (20-50%)
                    search_progress = 20 + (progress * 0.3)  # 20% to 50%
                    logger.set_progress(int(search_progress))
        
        if logger:
            logger.log_success(f"Embedding complete: {total_cached} from cache, {total_new} computed")
        
        return np.array(embeddings)
    
    def prepare_paper_text(self, entry: Dict[str, Any]) -> str:
        """Prepare text for embedding from a paper entry.
        
        Args:
            entry: Paper entry dictionary
            
        Returns:
            Combined text for embedding
        """
        parts = []
        
        # Add title (most important)
        title = entry.get('title', '').strip()
        if title:
            parts.append(title)
        
        # Add abstract (very important for topic matching)
        abstract = entry.get('abstract', '').strip()
        if abstract:
            parts.append(abstract)
        
        # Add tags if available (already topic-focused)
        tags = entry.get('tags', '').strip()
        if tags:
            parts.append(tags)
        
        # Add keywords if available
        keywords = entry.get('keywords', '').strip()
        if keywords:
            parts.append(keywords)
        
        return ' '.join(parts)
    
    def exact_search(self, query: str, df: pd.DataFrame, logger=None) -> List[Tuple[int, float]]:
        """Perform exact case-insensitive search.
        
        Args:
            query: Search query
            df: DataFrame with paper data
            logger: Optional logger for progress tracking
            
        Returns:
            List of (index, score) tuples where score is 1.0 for exact matches
        """
        if logger:
            logger.log_info(f"Starting exact search for '{query}' across {len(df)} papers")
        
        query_lower = query.lower()
        results = []
        
        for idx, row in df.iterrows():
            text_fields = [
                str(row.get('title', '')),
                str(row.get('abstract', '')),
                str(row.get('tags', '')),
                str(row.get('keywords', ''))
            ]
            
            combined_text = ' '.join(text_fields).lower()
            
            if query_lower in combined_text:
                results.append((idx, 1.0))
                if logger and len(results) <= 3:
                    title = str(row.get('title', 'No title'))[:50]
                    logger.log_info(f"Exact match found: '{title}...'")
        
        if logger:
            logger.log_success(f"Exact search complete: {len(results)} matches found")
        
        return results
    
    def fuzzy_search(self, query: str, df: pd.DataFrame, threshold: float = 80.0, logger=None) -> List[Tuple[int, float]]:
        """Perform fuzzy search using edit distance.
        
        Args:
            query: Search query
            df: DataFrame with paper data
            threshold: Minimum similarity score (0-100)
            logger: Optional logger for progress tracking
            
        Returns:
            List of (index, score) tuples where score is normalized (0-1)
        """
        if logger:
            logger.log_info(f"Starting fuzzy search for '{query}' (threshold: {threshold}%)")
        
        results = []
        processed = 0
        
        for idx, row in df.iterrows():
            text_fields = [
                str(row.get('title', '')),
                str(row.get('abstract', '')),
                str(row.get('tags', ''))
            ]
            
            max_score = 0
            for field in text_fields:
                if field:
                    score = fuzz.partial_ratio(query.lower(), field.lower())
                    max_score = max(max_score, score)
            
            if max_score >= threshold:
                results.append((idx, max_score / 100.0))
                if logger and len(results) <= 3:
                    title = str(row.get('title', 'No title'))[:50]
                    logger.log_info(f"Fuzzy match ({max_score}%): '{title}...'")
            
            processed += 1
            if logger and processed % 100 == 0:
                logger.log_info(f"Fuzzy search progress: {processed}/{len(df)} papers processed")
        
        if logger:
            logger.log_success(f"Fuzzy search complete: {len(results)} matches found above {threshold}% threshold")
        
        return results
    
    def semantic_search(self, query: str, df: pd.DataFrame, threshold: float = 0.7, logger=None) -> List[Tuple[int, float]]:
        """Perform semantic search using embeddings.
        
        Args:
            query: Search query
            df: DataFrame with paper data
            threshold: Minimum similarity score (0-1)
            logger: Optional logger for progress tracking
            
        Returns:
            List of (index, score) tuples
        """
        if logger:
            logger.log_info(f"Starting semantic search for '{query}' (threshold: {threshold:.2f})")
            logger.log_info(f"Computing query embedding...")
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        if logger:
            logger.log_info(f"Preparing text from {len(df)} papers...")
        
        # Prepare paper texts and get embeddings
        paper_texts = [self.prepare_paper_text(row.to_dict()) for _, row in df.iterrows()]
        
        # Count papers with meaningful text
        non_empty_texts = sum(1 for text in paper_texts if text.strip())
        if logger:
            logger.log_info(f"Found {non_empty_texts} papers with content for embedding")
            logger.log_info("Computing paper embeddings (checking cache first)...")
        
        paper_embeddings = self.get_embeddings_batch(paper_texts, logger=logger)
        
        if logger:
            logger.log_info("Computing semantic similarities...")
        
        # Calculate similarities with numerical safety
        with np.errstate(divide='ignore', invalid='ignore'):
            similarities = cosine_similarity([query_embedding], paper_embeddings)[0]
            # Replace any invalid values (NaN, inf) with 0
            similarities = np.nan_to_num(similarities, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Filter by threshold and return results
        results = []
        for idx, similarity in enumerate(similarities):
            if similarity >= threshold:
                results.append((idx, float(similarity)))
                if logger and len(results) <= 3:
                    title = str(df.iloc[idx].get('title', 'No title'))[:50]
                    logger.log_info(f"Semantic match ({similarity:.3f}): '{title}...'")
        
        if logger:
            logger.log_success(f"Semantic search complete: {len(results)} matches found above {threshold:.2f} threshold")
        
        return results
    
    def multi_search(
        self, 
        query: str, 
        df: pd.DataFrame,
        methods: List[str] = None,
        exact_weight: float = 1.0,
        fuzzy_weight: float = 0.8,
        semantic_weight: float = 1.0,
        fuzzy_threshold: float = 80.0,
        semantic_threshold: float = 0.7,
        max_results: int = 50,
        logger=None
    ) -> pd.DataFrame:
        """Perform multi-level search combining exact, fuzzy, and semantic methods.
        
        Args:
            query: Search query
            df: DataFrame with paper data
            methods: List of methods to use ('exact', 'fuzzy', 'semantic')
            exact_weight: Weight for exact matches
            fuzzy_weight: Weight for fuzzy matches
            semantic_weight: Weight for semantic matches
            fuzzy_threshold: Minimum score for fuzzy matches (0-100)
            semantic_threshold: Minimum score for semantic matches (0-1)
            max_results: Maximum number of results to return
            logger: Optional logger for progress tracking
            
        Returns:
            DataFrame with search results including relevance scores
        """
        if methods is None:
            methods = ['exact', 'fuzzy', 'semantic']
        
        if logger:
            logger.log_info(f"Starting multi-search for '{query}' using methods: {', '.join(methods)}")
            logger.log_info(f"Dataset contains {len(df)} papers")
        
        # Store all results with their method and score
        all_results = {}
        
        # Exact search
        if 'exact' in methods:
            exact_results = self.exact_search(query, df, logger=logger)
            for idx, score in exact_results:
                if idx not in all_results:
                    all_results[idx] = {'exact': 0, 'fuzzy': 0, 'semantic': 0}
                all_results[idx]['exact'] = score
        
        # Fuzzy search
        if 'fuzzy' in methods:
            fuzzy_results = self.fuzzy_search(query, df, threshold=fuzzy_threshold, logger=logger)
            for idx, score in fuzzy_results:
                if idx not in all_results:
                    all_results[idx] = {'exact': 0, 'fuzzy': 0, 'semantic': 0}
                all_results[idx]['fuzzy'] = score
        
        # Semantic search
        if 'semantic' in methods:
            semantic_results = self.semantic_search(query, df, threshold=semantic_threshold, logger=logger)
            for idx, score in semantic_results:
                if idx not in all_results:
                    all_results[idx] = {'exact': 0, 'fuzzy': 0, 'semantic': 0}
                all_results[idx]['semantic'] = score
        
        if logger:
            logger.log_info("Combining and ranking results...")
        
        # Combine scores and create results DataFrame
        results_data = []
        for idx, scores in all_results.items():
            combined_score = (
                scores['exact'] * exact_weight +
                scores['fuzzy'] * fuzzy_weight +
                scores['semantic'] * semantic_weight
            ) / (exact_weight + fuzzy_weight + semantic_weight)
            
            # Get the original paper data
            paper_data = df.iloc[idx].to_dict()
            paper_data['search_score'] = combined_score
            paper_data['exact_score'] = scores['exact']
            paper_data['fuzzy_score'] = scores['fuzzy']
            paper_data['semantic_score'] = scores['semantic']
            paper_data['original_index'] = idx
            
            results_data.append(paper_data)
        
        # Convert to DataFrame and sort by combined score
        results_df = pd.DataFrame(results_data)
        if not results_df.empty:
            results_df = results_df.sort_values('search_score', ascending=False)
            results_df = results_df.head(max_results)
            results_df = results_df.reset_index(drop=True)
            
            if logger:
                logger.log_success(f"Search complete! Found {len(results_df)} total results (showing top {min(max_results, len(results_df))})")
                
                # Show method breakdown
                exact_count = sum(1 for _, row in results_df.iterrows() if row['exact_score'] > 0)
                fuzzy_count = sum(1 for _, row in results_df.iterrows() if row['fuzzy_score'] > 0)
                semantic_count = sum(1 for _, row in results_df.iterrows() if row['semantic_score'] > 0)
                
                logger.log_info(f"Results breakdown: {exact_count} exact, {fuzzy_count} fuzzy, {semantic_count} semantic matches")
        else:
            if logger:
                logger.log_info("No results found matching the search criteria")
        
        return results_df


class HybridSemanticSearcher(SemanticSearcher):
    """Enhanced semantic search combining embeddings with LLM analysis."""
    
    def __init__(self, api_key: Optional[str] = None, embedding_model: str = "text-embedding-3-small", llm_model: str = "gpt-4o-mini"):
        """Initialize the hybrid searcher.
        
        Args:
            api_key: OpenAI API key (will use environment variable if not provided)
            embedding_model: OpenAI embedding model to use
            llm_model: OpenAI LLM model for relevance analysis (gpt-4o-mini is cheapest and works great)
        """
        super().__init__(api_key, embedding_model)
        self.llm_model = llm_model
        self.llm_cache_dir = Path(".llm_relevance_cache")
        self.llm_cache_dir.mkdir(exist_ok=True)
    
    def _get_llm_cache_key(self, query: str, paper_content: str) -> str:
        """Generate cache key for LLM relevance analysis."""
        combined = f"{self.llm_model}:{query}:{paper_content[:200]}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _load_llm_cache(self, cache_key: str) -> Optional[Dict]:
        """Load LLM analysis from cache."""
        cache_file = self.llm_cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load LLM cache file {cache_file}: {e}")
        return None
    
    def _save_llm_cache(self, cache_key: str, analysis: Dict) -> None:
        """Save LLM analysis to cache."""
        cache_file = self.llm_cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(analysis, f)
        except Exception as e:
            logger.warning(f"Failed to save LLM cache file {cache_file}: {e}")
    
    def llm_analyze_relevance(self, query: str, papers: List[Dict], logger=None) -> List[Dict]:
        """Use LLM to analyze paper relevance with reasoning.
        
        Args:
            query: Search query
            papers: List of paper dictionaries
            logger: Optional logger for progress tracking
            
        Returns:
            List of papers with added LLM analysis
        """
        if logger:
            logger.log_info(f"💰 Starting GPT-{self.llm_model} analysis for {len(papers)} papers")
            logger.log_info(f"🔍 Query: '{query}'")
            logger.log_info(f"💾 Checking cache first to minimize API costs...")
        
        analyzed_papers = []
        cache_hits = 0
        llm_calls = 0
        
        for i, paper in enumerate(papers):
            # Update progress
            if logger and hasattr(logger, 'set_progress'):
                # Map LLM analysis progress (50-90% of overall search)
                progress = (i / len(papers)) * 40 + 50  # 50% to 90%
                logger.set_progress(int(progress))
            
            # Prepare paper content for analysis
            title = paper.get('title', 'No title')
            abstract = paper.get('abstract', '')[:500]  # Limit to 500 chars for cost
            paper_content = f"{title}\n{abstract}"
            
            # Check cache first
            cache_key = self._get_llm_cache_key(query, paper_content)
            cached_analysis = self._load_llm_cache(cache_key)
            
            if cached_analysis:
                paper_with_analysis = paper.copy()
                paper_with_analysis.update(cached_analysis)
                analyzed_papers.append(paper_with_analysis)
                cache_hits += 1
                if logger and cache_hits <= 3:  # Only log first few cache hits to avoid spam
                    logger.log_info(f"💾 CACHE HIT #{cache_hits}: '{title[:40]}...' (saved API cost)")
                continue
            
            # LLM analysis prompt
            prompt = f"""Analyze this research paper's relevance to the query.

Query: "{query}"

Paper:
Title: {title}
Abstract: {abstract}

Provide analysis in this exact JSON format:
{{
  "llm_relevance_score": 0-10,
  "llm_confidence": 0-10,
  "llm_reasoning": "1-2 sentence explanation",
  "llm_key_concepts": ["concept1", "concept2", "concept3"]
}}

Focus on:
1. Semantic similarity to query topic
2. Research relevance and quality
3. Conceptual overlap

Be precise and concise."""

            try:
                if logger:
                    logger.log_info(f"🌐 API CALL #{llm_calls + 1}: Analyzing '{title[:40]}...' with GPT-{self.llm_model}")
                
                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,  # Low temperature for consistent scoring
                    max_tokens=200   # Keep costs low
                )
                
                if logger:
                    logger.log_info(f"✅ API call #{llm_calls + 1} completed successfully")
                
                # Parse JSON response
                analysis_text = response.choices[0].message.content.strip()
                
                # Try to extract JSON from response
                try:
                    import re
                    json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
                    if json_match:
                        analysis = json.loads(json_match.group())
                    else:
                        raise ValueError("No JSON found in response")
                except (json.JSONDecodeError, ValueError):
                    # Fallback if JSON parsing fails
                    logger.log_error(f"Failed to parse LLM response: {analysis_text[:100]}...")
                    analysis = {
                        "llm_relevance_score": 5.0,
                        "llm_confidence": 3.0,
                        "llm_reasoning": "Analysis failed - using default score",
                        "llm_key_concepts": []
                    }
                
                # Validate and normalize scores
                analysis["llm_relevance_score"] = max(0, min(10, float(analysis.get("llm_relevance_score", 5))))
                analysis["llm_confidence"] = max(0, min(10, float(analysis.get("llm_confidence", 5))))
                
                # Cache the analysis
                self._save_llm_cache(cache_key, analysis)
                
                # Add analysis to paper
                paper_with_analysis = paper.copy()
                paper_with_analysis.update(analysis)
                analyzed_papers.append(paper_with_analysis)
                
                llm_calls += 1
                
            except Exception as e:
                if logger:
                    logger.log_error(f"LLM analysis failed for paper: {e}")
                
                # Add paper with default analysis
                paper_with_analysis = paper.copy()
                paper_with_analysis.update({
                    "llm_relevance_score": 5.0,
                    "llm_confidence": 1.0,
                    "llm_reasoning": f"Analysis failed: {str(e)[:50]}",
                    "llm_key_concepts": []
                })
                analyzed_papers.append(paper_with_analysis)
        
        if logger:
            total_cost_estimate = llm_calls * 0.0005  # Rough estimate for gpt-4o-mini
            logger.log_success(f"💰 LLM ANALYSIS COMPLETE:")
            logger.log_info(f"📊 Total papers analyzed: {len(papers)}")
            logger.log_info(f"💾 Cache hits: {cache_hits} (saved ${cache_hits * 0.0005:.3f})")
            logger.log_info(f"🌐 New API calls: {llm_calls} (cost ~${total_cost_estimate:.3f})")
            if cache_hits > llm_calls:
                logger.log_info(f"🎉 Cache saved you {(cache_hits / (cache_hits + llm_calls)) * 100:.1f}% on API costs!")
        
        return analyzed_papers
    
    def hybrid_search(self, query: str, df: pd.DataFrame, threshold: float = 0.6, 
                     max_embedding_candidates: int = 50, max_results: int = 20, logger=None) -> List[Tuple[int, float]]:
        """Perform hybrid search: embeddings for speed + LLM for precision.
        
        Args:
            query: Search query
            df: DataFrame with paper data
            threshold: Minimum embedding similarity for initial filter
            max_embedding_candidates: Max papers to pass to LLM analysis
            max_results: Maximum final results to return
            logger: Optional logger for progress tracking
            
        Returns:
            List of (index, combined_score) tuples
        """
        if logger:
            logger.log_info(f"🚀 HYBRID SEARCH: '{query}' across {len(df)} papers")
            logger.log_info(f"📊 Phase 1: Fast embedding scan of ALL {len(df)} papers (threshold: {threshold:.2f})")
        
        # Phase 1: Fast embedding search to get candidates
        embedding_results = self.semantic_search(query, df, threshold=threshold, logger=logger)
        
        if not embedding_results:
            if logger:
                logger.log_info("⚠️ No embedding candidates found - lowering threshold to find some results")
            # Try with lower threshold if no results
            embedding_results = self.semantic_search(query, df, threshold=threshold * 0.7, logger=logger)
        
        # Limit candidates for LLM analysis (cost control)
        original_candidates = len(embedding_results)
        embedding_results = embedding_results[:max_embedding_candidates]
        
        if not embedding_results:
            if logger:
                logger.log_info("❌ No candidates found even with lower threshold")
            return []
        
        if logger:
            if original_candidates > max_embedding_candidates:
                logger.log_info(f"📋 Embedding filter: {original_candidates} candidates found, limiting to top {len(embedding_results)} for cost control")
            else:
                logger.log_info(f"📋 Embedding filter: {len(embedding_results)} candidates found")
            logger.log_info(f"🤖 Phase 2: GPT-{self.llm_model} analysis of {len(embedding_results)} papers (this will take time and cost money)")
        
        # Phase 2: Convert results to paper dictionaries for LLM analysis
        candidate_papers = []
        for idx, embedding_score in embedding_results:
            paper = df.iloc[idx].to_dict()
            paper['original_index'] = idx
            paper['embedding_score'] = embedding_score
            candidate_papers.append(paper)
        
        # LLM analysis of candidates
        analyzed_papers = self.llm_analyze_relevance(query, candidate_papers, logger=logger)
        
        # Store analyzed papers for dashboard access
        self._last_analyzed_papers = analyzed_papers
        
        # Phase 3: Combine scores and rank
        if logger:
            logger.log_info("Phase 3: Combining embedding + LLM scores")
        
        final_results = []
        for paper in analyzed_papers:
            embedding_score = paper['embedding_score']
            llm_score = paper['llm_relevance_score'] / 10.0  # Normalize to 0-1
            confidence = paper['llm_confidence'] / 10.0
            
            # Weighted combination: embedding (40%) + LLM (60%), adjusted by confidence
            combined_score = (
                0.4 * embedding_score + 
                0.6 * llm_score * confidence + 
                0.1 * confidence  # Bonus for high confidence
            )
            
            final_results.append((paper['original_index'], combined_score))
        
        # Sort by combined score and limit results
        final_results.sort(key=lambda x: x[1], reverse=True)
        final_results = final_results[:max_results]
        
        if logger:
            logger.log_success(f"🎯 HYBRID SEARCH COMPLETE!")
            logger.log_info(f"📋 Phase 1: Scanned {len(df)} papers with embeddings")
            logger.log_info(f"🤖 Phase 2: GPT analyzed {len(embedding_results)} top candidates")
            logger.log_info(f"🏆 Phase 3: Returning {len(final_results)} best results")
            
            # Log top results for debugging
            logger.log_info(f"🥇 TOP RESULTS:")
            for i, (idx, score) in enumerate(final_results[:3]):
                title = str(df.iloc[idx].get('title', 'No title'))[:50]
                logger.log_info(f"  #{i+1}: {score:.3f} - '{title}...'")
        
        return final_results
    
    def llm_only_search(self, query: str, df: pd.DataFrame, max_results: int = 20, 
                       relevance_threshold: float = 6.0, logger=None) -> List[Tuple[int, float]]:
        """Perform LLM-only search: analyze all papers with GPT for maximum quality.
        
        Args:
            query: Search query
            df: DataFrame with paper data
            max_results: Maximum final results to return
            relevance_threshold: Minimum LLM relevance score (0-10) to include
            logger: Optional logger for progress tracking
            
        Returns:
            List of (index, llm_score) tuples
        """
        if logger:
            logger.log_info(f"🤖 LLM-ONLY SEARCH: '{query}' with GPT-{self.llm_model}")
            logger.log_info(f"📊 Will analyze ALL {len(df)} papers (no embedding filter)")
            logger.log_info(f"⚠️ This will be expensive but highest quality!")
            
            # Cost estimate
            estimated_cost = len(df) * 0.0005  # Rough estimate
            logger.log_info(f"💰 Estimated cost: ~${estimated_cost:.2f}")
        
        # Convert all papers to list for LLM analysis
        all_papers = []
        for idx, row in df.iterrows():
            paper = row.to_dict()
            paper['original_index'] = idx
            all_papers.append(paper)
        
        # Analyze ALL papers with LLM
        analyzed_papers = self.llm_analyze_relevance(query, all_papers, logger=logger)
        
        # Filter by relevance threshold and sort
        results = []
        for paper in analyzed_papers:
            llm_score = paper.get('llm_relevance_score', 0)
            if llm_score >= relevance_threshold:
                # Normalize score to 0-1 range for consistency
                normalized_score = llm_score / 10.0
                results.append((paper['original_index'], normalized_score))
        
        # Sort by LLM score and limit results
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:max_results]
        
        if logger:
            logger.log_success(f"🎯 LLM-ONLY SEARCH COMPLETE!")
            logger.log_info(f"🤖 Analyzed ALL {len(df)} papers with GPT")
            logger.log_info(f"🏆 Found {len(results)} papers above {relevance_threshold}/10 threshold")
            
            # Log top results
            if results:
                logger.log_info(f"🥇 TOP RESULTS:")
                for i, (idx, score) in enumerate(results[:3]):
                    title = str(df.iloc[idx].get('title', 'No title'))[:50]
                    llm_score = score * 10  # Convert back to 0-10 scale for display
                    logger.log_info(f"  #{i+1}: {llm_score:.1f}/10 - '{title}...'")
        
        return results
    
    def expand_query_with_llm(self, query: str, logger=None) -> List[str]:
        """Use LLM to expand query with related terms.
        
        Args:
            query: Original search query
            logger: Optional logger for progress tracking
            
        Returns:
            List of expanded query terms including the original
        """
        if logger:
            logger.log_info(f"Expanding query '{query}' with LLM")
        
        prompt = f"""Research Query: "{query}"

Generate 5-8 related terms, synonyms, and alternative phrases that researchers might use for this topic. Include:
1. Technical terminology
2. Alternative names/acronyms
3. Related concepts
4. Common variations

Respond with a JSON list: ["term1", "term2", "term3", ...]

Be concise and research-focused."""

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON list
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                expanded_terms = json.loads(json_match.group())
                # Add original query at the beginning
                all_terms = [query] + [term for term in expanded_terms if term.lower() != query.lower()]
                
                if logger:
                    logger.log_success(f"Query expanded to {len(all_terms)} terms: {', '.join(all_terms[:3])}...")
                
                return all_terms
                
        except Exception as e:
            if logger:
                logger.log_error(f"Query expansion failed: {e}")
        
        # Fallback: return original query
        return [query]


def search_papers(
    query: str,
    input_file: str,
    methods: List[str] = None,
    semantic_threshold: float = 0.7,
    fuzzy_threshold: float = 80.0,
    max_results: int = 50,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """High-level function to search papers in a bibliography file.
    
    Args:
        query: Search query
        input_file: Path to CSV/Excel file with paper data
        methods: Search methods to use
        semantic_threshold: Threshold for semantic similarity (0-1)
        fuzzy_threshold: Threshold for fuzzy matching (0-100)
        max_results: Maximum number of results
        output_file: Optional path to save results
        
    Returns:
        DataFrame with search results
    """
    # Load the data
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    elif input_file.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(input_file)
    else:
        raise ValueError("Input file must be CSV or Excel format")
    
    # Initialize searcher
    searcher = SemanticSearcher()
    # Perform search
    results = searcher.multi_search(
        query=query,
        df=df,
        methods=methods,
        semantic_threshold=semantic_threshold,
        fuzzy_threshold=fuzzy_threshold,
        max_results=max_results
    )
    
    # Save results if requested
    if output_file and not results.empty:
        if output_file.endswith('.csv'):
            results.to_csv(output_file, index=False)
        elif output_file.endswith(('.xlsx', '.xls')):
            results.to_excel(output_file, index=False)
        logger.info(f"Search results saved to {output_file}")
    
    return results