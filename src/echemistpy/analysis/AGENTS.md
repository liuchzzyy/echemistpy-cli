# ANALYSIS MODULE KNOWLEDGE BASE

## OVERVIEW
Provides a unified analysis pipeline organized by scientific domain. Analyzers are registered by technique and instrument.

## STRUCTURE
```
src/echemistpy/analysis/
├── registry.py           # TechniqueRegistry & TechniqueAnalyzer base
├── pipeline.py           # AnalysisPipeline orchestrator
├── xas/                  # XAS Domain
│   ├── analyzer.py       # XASAnalyzer
│   ├── processing.py     # Preprocessing (calib, align, deglitch)
│   ├── fitting.py        # Math (PCA, LCF)
│   └── plotting.py       # Visualization
├── echem/                # Electrochemistry Domain
│   └── analyzer.py       # GalvanostaticAnalyzer
└── stxm/                 # STXM Domain
    └── analyzer.py       # STXMAnalyzer
```

## ARCHITECTURE
`AnalysisPipeline` -> `TechniqueRegistry` -> `TechniqueAnalyzer` -> `analyze()` -> `_compute()`

## EXTENSION GUIDE
To add a new analysis (e.g., `NewMethod`):

1.  **Choose Domain**: `src/echemistpy/analysis/{domain}/` (create if new)
2.  **Create File**: `src/echemistpy/analysis/{domain}/analyzer.py`
3.  **Inherit**: `from ..registry import TechniqueAnalyzer`
4.  **Define**:
    - `technique = "new_method"`
    - `required_columns = ("col1", "col2")`
5.  **Implement**:
    - `_compute(raw_data)`: Return `(AnalysisData, AnalysisDataInfo)`
6.  **Register**: Add to `create_default_registry()` in `registry.py`.

## KEY CLASSES
- **`TechniqueAnalyzer`**: Base class. Handles validation, preprocessing, and metadata inheritance.
- **`TechniqueRegistry`**: Stores available analyzers.
- **`AnalysisData`**: Container for results.

## CONVENTIONS
- **Domain Cohesion**: Keep all logic for a technique (processing, plotting, fitting) in its domain folder.
- **Do Not Override `analyze()`**: Override `_compute()` instead.
- **Stateless**: Analyzers should be stateless regarding data.
