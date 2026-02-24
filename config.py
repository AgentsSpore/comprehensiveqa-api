"""Configuration for ComprehensiveQA API"""

from typing import Dict, Any


class Config:
    """Global configuration settings"""
    
    # RVR Algorithm Settings
    DEFAULT_MAX_ROUNDS = 3
    DEFAULT_DOCS_PER_ROUND = 5
    DEFAULT_VERIFICATION_THRESHOLD = 0.6
    
    # Model Settings
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Lightweight sentence transformer
    
    # API Settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # Limits
    MAX_ROUNDS_LIMIT = 5
    MAX_DOCS_PER_ROUND = 10
    MIN_VERIFICATION_THRESHOLD = 0.0
    MAX_VERIFICATION_THRESHOLD = 1.0
    
    @classmethod
    def get_rvr_defaults(cls) -> Dict[str, Any]:
        return {
            "max_rounds": cls.DEFAULT_MAX_ROUNDS,
            "docs_per_round": cls.DEFAULT_DOCS_PER_ROUND,
            "verification_threshold": cls.DEFAULT_VERIFICATION_THRESHOLD
        }


# Domain-specific configurations
DOMAIN_CONFIG = {
    "medical": {
        "verification_threshold": 0.7,  # Higher threshold for medical accuracy
        "max_rounds": 4
    },
    "legal": {
        "verification_threshold": 0.65,  # Precision important for legal
        "max_rounds": 4
    },
    "research": {
        "verification_threshold": 0.6,  # Balance breadth and relevance
        "max_rounds": 3
    },
    "general": {
        "verification_threshold": 0.6,
        "max_rounds": 3
    }
}