"""Constants for the ArXiv search application."""

# ArXiv API
BASE_URL = "https://oaipmh.arxiv.org/oai"

# Cohere models
EMBED_MODEL = "embed-v4.0"
RERANK_MODEL = "rerank-v3.5"

# Default values
DEFAULT_SET_SPEC = "cs"
DEFAULT_METADATA_PREFIX = "arXiv"
DEFAULT_TOP_K = 10
DEFAULT_RETRIEVAL_K = 100
DEFAULT_STRATEGY = "embed-rerank"

# Strategy options
STRATEGY_EMBED_RERANK = "embed-rerank"
STRATEGY_RERANK_ONLY = "rerank-only"
VALID_STRATEGIES = [STRATEGY_EMBED_RERANK, STRATEGY_RERANK_ONLY]

# Date presets
PRESET_YESTERDAY = "yesterday"
PRESET_LAST_WEEK = "last-week"
VALID_PRESETS = [PRESET_YESTERDAY, PRESET_LAST_WEEK]

# OAI-PMH namespaces
OAI_NAMESPACES = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "arxiv": "http://arxiv.org/OAI/arXiv/",
}
