# ArXiv Semantic Search

A command-line tool for searching arXiv papers using natural language queries with semantic search powered by Cohere embeddings and reranking.

## Features

- = **Natural Language Search**: Query papers using plain English like "papers about information retrieval and search engines that use neural networks"
- >à **Semantic Understanding**: Uses Cohere Embed v4 and Rerank 3.5 for intelligent paper matching
- =Ê **Rich Output**: Beautiful table formatting with relevance scores, authors, and links
- =Ó **Flexible Date Ranges**: Search recent papers (yesterday, last week) or custom date ranges
- =Ä **Multiple Output Formats**: Human-readable tables or JSON for programmatic use
- <¯ **Multiple arXiv Categories**: Support for Computer Science, Math, Physics, and more

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd arxiv-search

# Install dependencies
uv sync

# Set up your Cohere API key
export COHERE_API_KEY="your-cohere-api-key"
```

## Usage

### Semantic Search (New!)

Search for papers using natural language queries:

```bash
# Search for recent papers about neural networks and information retrieval
uv run arxiv-search search "papers about information retrieval and search engines that use neural networks" --preset yesterday

# Search with custom date range and show abstracts
uv run arxiv-search search "transformer architectures for computer vision" --from-date 2024-01-01 --until-date 2024-01-07 --abstract

# Get top 5 results in JSON format
uv run arxiv-search search "reinforcement learning for robotics" --preset last-week --top-k 5 --json

# Search in other arXiv categories (e.g., Mathematics)
uv run arxiv-search search "topology and algebraic geometry" --preset yesterday --set math
```

### Raw Metadata Harvest (Original functionality)

For raw JSON output without semantic search:

```bash
# Harvest raw metadata for yesterday
uv run arxiv-search harvest --preset yesterday

# Custom date range
uv run arxiv-search harvest --from-date 2024-01-01 --until-date 2024-01-07
```

## Examples

### Example 1: Information Retrieval Research
```bash
uv run arxiv-search search "information retrieval using neural networks and transformers" --preset last-week --top-k 5 --abstract
```

### Example 2: Computer Vision Papers
```bash
uv run arxiv-search search "object detection and computer vision with deep learning" --from-date 2024-01-01 --until-date 2024-01-31
```

### Example 3: Machine Learning Theory
```bash
uv run arxiv-search search "theoretical foundations of deep learning and optimization" --preset yesterday --json
```

## Output Format

The tool provides rich, formatted output showing:
- **Rank**: Paper ranking by relevance
- **Score**: Semantic relevance score (0-1)
- **Title**: Paper title
- **Authors**: Author list (truncated if long)
- **arXiv ID**: For accessing the paper
- **Links**: Direct URLs to abstract and PDF

## API Key Setup

Get your free Cohere API key from [cohere.com](https://cohere.com) and set it as an environment variable:

```bash
export COHERE_API_KEY="your-api-key-here"
```

## Supported arXiv Categories

- `cs`: Computer Science (default)
- `math`: Mathematics  
- `physics`: Physics
- `q-bio`: Quantitative Biology
- `q-fin`: Quantitative Finance
- `stat`: Statistics

## Requirements

- Python 3.12+
- Cohere API key
- Internet connection for arXiv API and Cohere services