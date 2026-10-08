from abc import ABC, abstractmethod
from typing import Any

from sb.llm.models import ParseResult, PreprocessingConfig


class AnalysisProfile(ABC):
    id: str
    version: int
    task: str
    task_name: str
    primary_classification_field: str
    accepted_labels: tuple[str, ...]
    preprocessing: PreprocessingConfig
    prompt_versions: dict[str, int]

    @abstractmethod
    def render_prompt(self, strategy: str, source: str) -> str:
        pass

    @abstractmethod
    def validate_auxiliary(
        self, parsed: dict[str, Any], source_line_count: int
    ) -> tuple[dict[str, Any], list[str]]:
        pass

    def smartbugs_findings(self, result: ParseResult) -> list[dict[str, Any]]:
        return []
