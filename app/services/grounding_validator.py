import re
from typing import List
from app.models.analysis import ProviderReport, AnalysisResult
from app.core.exceptions import GroundingValidationError

class GroundingValidator:
    @staticmethod
    def validate(report: ProviderReport, results: List[AnalysisResult]) -> None:
        """Validates that all finding and recommendation IDs are formatted correctly,

        ensures all references link existing elements, and prevents orphan elements.
        """
        valid_result_ids = {res.result_id for res in results}

        # 1. Validate Findings
        finding_ids = set()
        for idx, finding in enumerate(report.findings):
            fid = finding.id
            if not fid or not fid.strip():
                raise GroundingValidationError(f"Finding at index {idx} has an empty ID.")

            if not re.match(r"^finding_\d+$", fid):
                raise GroundingValidationError(f"Finding ID '{fid}' has an invalid format. Must match 'finding_<number>'.")

            if fid in finding_ids:
                raise GroundingValidationError(f"Duplicate finding ID: '{fid}'. Finding IDs must be unique.")
            finding_ids.add(fid)

            # Ensure finding is not an orphan
            if not finding.evidence_refs:
                raise GroundingValidationError(f"Finding '{fid}' contains no evidence references. A finding must cite at least one analysis result.")

            # Validate evidence references
            for ref in finding.evidence_refs:
                if ref not in valid_result_ids:
                    raise GroundingValidationError(f"Finding '{fid}' references a non-existent or foreign result ID: '{ref}'.")

        # 2. Validate Recommendations
        recommendation_ids = set()
        for idx, rec in enumerate(report.recommendations):
            rid = rec.id
            if not rid or not rid.strip():
                raise GroundingValidationError(f"Recommendation at index {idx} has an empty ID.")

            if not re.match(r"^recommendation_\d+$", rid):
                raise GroundingValidationError(f"Recommendation ID '{rid}' has an invalid format. Must match 'recommendation_<number>'.")

            if rid in recommendation_ids:
                raise GroundingValidationError(f"Duplicate recommendation ID: '{rid}'. Recommendation IDs must be unique.")
            recommendation_ids.add(rid)

            # Ensure recommendation is not an orphan
            if not rec.finding_refs:
                raise GroundingValidationError(f"Recommendation '{rid}' contains no finding references. A recommendation must cite at least one finding.")

            # Validate finding references
            for f_ref in rec.finding_refs:
                if f_ref not in finding_ids:
                    raise GroundingValidationError(f"Recommendation '{rid}' references a non-existent or foreign finding ID: '{f_ref}'.")

        # Verify that there is at least one limitation
        if not report.limitations:
            raise GroundingValidationError("The report must contain at least one limitation statement.")
