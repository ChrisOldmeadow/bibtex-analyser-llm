"""Word cloud visualization using wordcloud2.js for interactive word clouds."""
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from jinja2 import Template

# Template for the interactive word cloud
WORDCLOUD_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>Interactive Word Cloud</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/wordcloud2.js/1.2.2/wordcloud2.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        #wordcloud-container { 
            width: 1200px; 
            height: 800px; 
            margin: 0 auto;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }
        #wordcloud { width: 100%; height: 100%; }
        .wordcloud-tooltip {
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 14px;
            pointer-events: none;
            z-index: 1000;
            max-width: 300px;
            display: none;
        }
        .wordcloud-word {
            cursor: pointer;
            transition: all 0.2s;
        }
        .wordcloud-word:hover {
            color: #1f77b4 !important;
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <h1 style="text-align: center;">Interactive Word Cloud</h1>
    <div id="wordcloud-container">
        <div id="wordcloud"></div>
    </div>
    <div id="tooltip" class="wordcloud-tooltip"></div>
    
    <script>
        // Word data
        const wordData = {{ words|tojson|safe }};
        
        // Prepare data for wordcloud2
        const wordList = [];
        const tagToPapers = {};
        
        wordData.forEach(item => {
            wordList.push([item.text, item.size]);
            if (item.papers) {
                tagToPapers[item.text] = item.papers;
            }
        });
        
        // Tooltip element
        const tooltip = document.getElementById('tooltip');
        
        // Function to show tooltip
        function showTooltip(event, text, content) {
            tooltip.innerHTML = content;
            tooltip.style.display = 'block';
            updateTooltipPosition(event);
        }
        
        // Function to update tooltip position
        function updateTooltipPosition(event) {
            const x = event.clientX + 10;
            const y = event.clientY + 10;
            tooltip.style.left = `${x}px`;
            tooltip.style.top = `${y}px`;
        }
        
        // Function to hide tooltip
        function hideTooltip() {
            tooltip.style.display = 'none';
        }
        
        // Initialize word cloud
        document.addEventListener('DOMContentLoaded', function() {
            WordCloud(document.getElementById('wordcloud'), {
                list: wordList,
                gridSize: 10,
                weightFactor: 10,
                fontFamily: 'Arial, sans-serif',
                color: function (word, size) {
                    // Generate a nice color based on the word
                    let hash = 0;
                    for (let i = 0; i < word.length; i++) {
                        hash = word.charCodeAt(i) + ((hash << 5) - hash);
                    }
                    const hue = Math.abs(hash % 360);
                    return `hsl(${hue}, 70%, 50%)`;
                },
                rotateRatio: 0.5,
                rotationSteps: 2,
                backgroundColor: '#ffffff',
                minSize: 8,
                drawOutOfBound: false,
                click: function(item) {
                    const papers = tagToPapers[item[0]];
                    if (papers && papers.length > 0) {
                        const paperList = papers.map(p => 
                            `<div style="margin-bottom: 8px;">
                                <div><strong>${p.title || 'No title'}</strong></div>
                                <div>${p.authors ? p.authors.join(', ') : 'Unknown authors'}</div>
                                <div>${p.year || 'No year'}</div>
                                ${p.url ? `<div><a href="${p.url}" target="_blank">View Paper</a></div>` : ''}
                            </div>`
                        ).join('');
                        
                        const content = `
                            <h3 style="margin: 0 0 10px 0; color: #1f77b4;">${item[0]}</h3>
                            <div style="max-height: 300px; overflow-y: auto;">
                                ${paperList}
                            </div>
                        `;
                        
                        // Show a modal or tooltip with the papers
                        const modal = document.createElement('div');
                        modal.style.position = 'fixed';
                        modal.style.top = '50%';
                        modal.style.left = '50%';
                        modal.style.transform = 'translate(-50%, -50%)';
                        modal.style.backgroundColor = 'white';
                        modal.style.padding = '20px';
                        modal.style.borderRadius = '8px';
                        modal.style.boxShadow = '0 4px 20px rgba(0,0,0,0.2)';
                        modal.style.zIndex = '1000';
                        modal.style.maxWidth = '600px';
                        modal.style.maxHeight = '80vh';
                        modal.style.overflow = 'auto';
                        
                        modal.innerHTML = `
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                <h2 style="margin: 0;">${item[0]}</h2>
                                <button id="close-modal" style="background: none; border: none; font-size: 20px; cursor: pointer;">×</button>
                            </div>
                            <div style="max-height: 60vh; overflow-y: auto;">
                                ${paperList}
                            </div>
                        `;
                        
                        // Close button functionality
                        modal.querySelector('#close-modal').addEventListener('click', () => {
                            document.body.removeChild(modal);
                            document.body.style.overflow = '';
                        });
                        
                        // Add overlay
                        const overlay = document.createElement('div');
                        overlay.style.position = 'fixed';
                        overlay.style.top = '0';
                        overlay.style.left = '0';
                        overlay.style.width = '100%';
                        overlay.style.height = '100%';
                        overlay.style.backgroundColor = 'rgba(0,0,0,0.5)';
                        overlay.style.zIndex = '999';
                        overlay.onclick = () => {
                            document.body.removeChild(overlay);
                            document.body.removeChild(modal);
                            document.body.style.overflow = '';
                        };
                        
                        // Prevent body scroll when modal is open
                        document.body.style.overflow = 'hidden';
                        
                        document.body.appendChild(overlay);
                        document.body.appendChild(modal);
                    }
                },
                hover: function(item, dimension, event) {
                    if (!dimension) {
                        hideTooltip();
                        return;
                    }
                    
                    const papers = tagToPapers[item[0]] || [];
                    const count = papers.length;
                    const paperCountText = count === 1 ? '1 paper' : `${count} papers`;
                    
                    showTooltip(event, item[0], `
                        <div><strong>${item[0]}</strong></div>
                        <div>${paperCountText}</div>
                        <div>Click to view papers</div>
                    `);
                },
                hover: hideTooltip
            });
            
            // Hide tooltip when mouse leaves the word cloud
            document.getElementById('wordcloud').addEventListener('mouseleave', hideTooltip);
        });
    </script>
</body>
</html>"""

def create_interactive_wordcloud(
    df: pd.DataFrame,
    output_file: str,
    tag_column: str = 'tags',
    title: str = 'Interactive Word Cloud',
    width: int = 1200,
    height: int = 800,
) -> None:
    """Create an interactive word cloud using wordcloud2.js.
    
    Args:
        df: DataFrame containing the data
        output_file: Path to save the HTML file
        tag_column: Name of the column containing tags
        title: Title for the word cloud
        width: Width of the word cloud in pixels
        height: Height of the word cloud in pixels
    """
    # Flatten tags and count frequencies
    all_tags = []
    for tags in df[tag_column].dropna():
        if isinstance(tags, str):
            all_tags.extend([tag.strip() for tag in tags.split(',')])
    
    tag_counts = Counter(all_tags)
    
    # Prepare word data for the word cloud
    words = []
    for tag, count in tag_counts.items():
        if not tag or tag.lower() == 'nan':
            continue
            
        # Get papers that have this tag
        papers = []
        for _, row in df[df[tag_column].str.contains(tag, na=False, case=False)].iterrows():
            papers.append({
                'title': row.get('title', ''),
                'authors': row.get('author', '').split(' and ') if 'author' in df.columns else [],
                'year': row.get('year', ''),
                'url': row.get('url', row.get('doi', ''))
            })
        
        words.append({
            'text': tag,
            'size': count * 10 + 10,  # Scale size for better visibility
            'papers': papers
        })
    
    # Sort by size (descending)
    words.sort(key=lambda x: x['size'], reverse=True)
    
    # Limit the number of words to improve performance
    words = words[:200]
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Render the template
    template = Template(WORDCLOUD_TEMPLATE)
    html_content = template.render(
        title=title,
        width=width,
        height=height,
        words=words
    )
    
    # Save the HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path
