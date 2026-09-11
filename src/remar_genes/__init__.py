"""
src/remar_genes/__init__.py
Pacote modular REMAR Genes — Espondiloartrites (PsA & AS)
"""

__version__ = "1.0.0"

from .annotator import FunctionalAnnotator, QualityAlert
from .modeling import (
    SwissModelHomologyModeler,
    RamachandranAnalyzer,
    generate_ramachandran_plotly_data,
    ModelingJobResult
)

__all__ = [
    "FunctionalAnnotator",
    "QualityAlert",
    "SwissModelHomologyModeler",
    "RamachandranAnalyzer",
    "generate_ramachandran_plotly_data",
    "ModelingJobResult"
]
