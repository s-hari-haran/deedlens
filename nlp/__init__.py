# NLP Module
from .ner_model import PropertyNERModel, extract_entities
from .embeddings import EmbeddingGenerator, generate_embeddings
# entity_resolution is available but not imported at module level
# to avoid torch/transformers import issues - import directly when needed
