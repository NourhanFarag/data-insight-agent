from abc import ABC, abstractmethod
from app.models.responses import DatasetSummary
from app.models.analysis import AnalysisPlan, ProviderReport, AnalysisResult

class BaseProvider(ABC):
    @abstractmethod
    async def create_analysis_plan(self, question: str, summary: DatasetSummary) -> AnalysisPlan:
        """Generates a structured analysis plan representing the operations to execute on the dataset."""
        pass

    @abstractmethod
    async def generate_report(self, question: str, summary: DatasetSummary, results: list[AnalysisResult]) -> ProviderReport:
        """Generates a final, evidence-grounded report containing findings and recommendations (no raw data)."""
        pass
