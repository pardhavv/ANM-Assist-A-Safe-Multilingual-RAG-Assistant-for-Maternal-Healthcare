"""
The Chunk dataclass lives in its own module (rather than inside ingest.py)
specifically so pickling/unpickling works correctly regardless of which
script is the entry point. If a class is defined inside a script that gets
run directly (`python -m app.retriever.ingest`), Python pickles it under the
module name `__main__`; unpickling later from a *different* entry point
(`run_query.py`, which is then `__main__` instead) fails because there's no
`Chunk` in that script's namespace. Keeping the class in a module that's
always imported via its full dotted path (`app.retriever.chunk_model`)
avoids this regardless of how ingestion or retrieval is invoked.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    doc_title: str
    chunk_index: int
    checksum: str
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
