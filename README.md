# ArXiv Semantic Search

A command-line tool for searching arXiv papers using natural language queries with semantic search powered by Cohere embeddings and reranking.

## Features

- **Natural Language Search**: Query papers using plain English like "papers about information retrieval and search engines that use neural networks"
- **Flexible Search Strategies**:
  - **Two-Stage (embed-rerank)**: Fast and cost-efficient (default)
    - Stage 1: Cohere Embed v4 to quickly filter top candidates via cosine similarity
    - Stage 2: Cohere Rerank 3.5 for precise final ranking
  - **Single-Stage (rerank-only)**: Maximum accuracy for critical searches
- **Rich Output**: Beautiful table formatting with relevance scores, authors, and links
- **Flexible Date Ranges**: Search recent papers (yesterday, last week) or custom date ranges
- **Multiple Output Formats**: Human-readable tables or JSON for programmatic use
- **Multiple arXiv Categories**: Support for Computer Science, Math, Physics, and more
- **Scalable**: Efficiently handles large document collections (1000+ papers)

## How It Works

Choose between two search strategies:

### Embed-Rerank (Default - Fast & Cost-Efficient)

1. **Retrieval Stage**: Uses Cohere Embed v4 to create embeddings and filter documents via cosine similarity
2. **Reranking Stage**: Uses Cohere Rerank 3.5 on the filtered candidates for final precision ranking

### Rerank-Only (Maximum Accuracy)

- **Single Stage**: Uses Cohere Rerank 3.5 directly on all documents for highest precision
- Best for critical searches where accuracy is more important than speed/cost

**Auto-optimization**: For small datasets (≤100 papers), the tool automatically uses rerank-only regardless of strategy choice.

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd arxivory

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
uv run arxivory search "papers about information retrieval and search engines that use neural networks" --preset yesterday

# Search with custom date range and show abstracts
uv run arxivory search "transformer architectures for computer vision" --from-date 2024-01-01 --until-date 2024-01-07 --abstract

# Get top 5 results in JSON format
uv run arxivory search "reinforcement learning for robotics" --preset last-week --top-k 5 --json

# Configure retrieval stage for large document sets
uv run arxivory search "deep learning optimization" --preset last-week --retrieval-k 200 --top-k 10

# Use rerank-only strategy for maximum accuracy
uv run arxivory search "critical medical AI research" --preset yesterday --strategy rerank-only

# Search in other arXiv categories (e.g., Mathematics)
uv run arxivory search "topology and algebraic geometry" --preset yesterday --set math
```

### Raw Metadata Harvest (Original functionality)

For raw JSON output without semantic search:

```bash
# Harvest raw metadata for yesterday
uv run arxivory harvest --preset yesterday

# Custom date range
uv run arxivory harvest --from-date 2024-01-01 --until-date 2024-01-07
```

## Examples

### Example 1: Information Retrieval Research

```bash
uv run arxivory search "information retrieval using neural networks and transformers" --preset last-week --top-k 5 --abstract
```

### Example 2: Computer Vision Papers

```bash
uv run arxivory search "object detection and computer vision with deep learning" --from-date 2024-01-01 --until-date 2024-01-31
```

### Example 3: Strategy Comparison

```bash
# Fast search with embed-rerank (default)
uv run arxivory search "theoretical foundations of deep learning" --preset last-week --strategy embed-rerank --top-k 10

# Maximum accuracy with rerank-only (slower, more expensive)
uv run arxivory search "theoretical foundations of deep learning" --preset last-week --strategy rerank-only --top-k 10
```

## Command Options

### Search Command

- `--preset`: Date shortcuts (`yesterday`, `last-week`)
- `--from-date` / `--until-date`: Custom date ranges (YYYY-MM-DD)
- `--top-k`: Final number of results to return (default: 10)
- `--strategy`: Search strategy (`embed-rerank` for speed, `rerank-only` for accuracy)
- `--retrieval-k`: Number of candidates to retrieve in embed-rerank stage 1 (default: 100)
- `--set`: arXiv category (default: `cs` for Computer Science)
- `--abstract`: Show paper abstracts in output
- `--json`: Output results as JSON

## Output Format

The tool provides rich, formatted output showing:

- **Rank**: Paper ranking by relevance
- **Score**: Semantic relevance score (0-1)
- **Title**: Paper title
- **Authors**: Author list (truncated if long)
- **arXiv ID**: For accessing the paper
- **Abstract**: Optional detailed abstract
- **Links**: Direct URLs to abstract and PDF

## Strategy Comparison

| Strategy                 | Speed  | Cost   | Accuracy  | Best For                          |
| ------------------------ | ------ | ------ | --------- | --------------------------------- |
| `embed-rerank` (default) | Fast   | Lower  | Very Good | General searches, large datasets  |
| `rerank-only`            | Slower | Higher | Maximum   | Critical searches, small datasets |

**Auto-optimization**: Small datasets (≤100 papers) automatically use rerank-only for optimal results.

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
