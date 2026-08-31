from dataclasses import dataclass, field

from brainforge.case import Case


@dataclass
class PipelineContext:
    case: Case
    rag_context: list = field(default_factory=list)
    previous_results: dict = field(default_factory=dict)
