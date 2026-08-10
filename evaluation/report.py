import os
import json
from datetime import datetime
from typing import List
from evaluation.models import EvaluationResult
from app.config import settings

def compile_metrics_markdown(results: List[EvaluationResult], provider: str, repetitions: int) -> str:
    """Aggregates scores and generates a readable Markdown report with target acceptance thresholds."""
    total = len(results)
    completed = sum(1 for r in results if r.error_category is None)

    schema_valid_cnt = sum(1 for r in results if r.planner_scores.schema_valid)
    plan_valid_cnt = sum(1 for r in results if r.planner_scores.plan_valid)

    # Repair counts
    plan_repair_attempted_cnt = sum(1 for r in results if r.planner_scores.plan_repair_attempted)
    plan_repair_succeeded_cnt = sum(1 for r in results if r.planner_scores.plan_repair_succeeded)
    initial_plan_valid_cnt = sum(1 for r in results if r.planner_scores.plan_valid and not r.planner_scores.plan_repair_attempted)

    avg_recall = sum(r.planner_scores.required_operation_recall for r in results) / total if total else 0.0
    avg_irrelevant = sum(r.planner_scores.irrelevant_operation_rate for r in results) / total if total else 0.0

    execution_passed_cnt = sum(1 for r in results if r.execution_passed)
    grounding_passed_cnt = sum(1 for r in results if r.grounding_scores.structurally_grounded)

    total_causal_flags = sum(r.grounding_scores.causal_claim_flags for r in results)
    total_numeric_flags = sum(r.grounding_scores.unsupported_numeric_claim_flags for r in results)

    success_cnt = sum(1 for r in results if r.final_success)

    latencies = [r.latency_ms for r in results if r.latency_ms is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    provider_failures = sum(1 for r in results if r.error_category == "provider_error")

    # Worst-case scenario calculation for multiple repetitions (by case success consistency)
    case_successes = {}
    for r in results:
        case_successes.setdefault(r.case_id, []).append(r.final_success)

    # Worst-case success rate: proportion of cases that succeeded in ALL of their repeated trials
    worst_case_success_cnt = sum(1 for cid, trials in case_successes.items() if all(trials))
    worst_case_success_rate = max(0.0, min(1.0, worst_case_success_cnt / len(case_successes) if case_successes else 0.0))

    schema_valid_rate = max(0.0, min(1.0, schema_valid_cnt / total if total else 0.0))
    plan_valid_rate = max(0.0, min(1.0, plan_valid_cnt / total if total else 0.0))
    initial_plan_valid_rate = max(0.0, min(1.0, initial_plan_valid_cnt / total if total else 0.0))
    post_repair_plan_valid_rate = plan_valid_rate
    plan_repair_rate = max(0.0, min(1.0, plan_repair_attempted_cnt / total if total else 0.0))
    repair_success_rate = max(0.0, min(1.0, plan_repair_succeeded_cnt / plan_repair_attempted_cnt if plan_repair_attempted_cnt > 0 else 0.0))

    execution_rate = max(0.0, min(1.0, execution_passed_cnt / total if total else 0.0))
    grounding_rate = max(0.0, min(1.0, grounding_passed_cnt / completed if completed else 0.0))
    end_to_end_rate = max(0.0, min(1.0, success_cnt / total if total else 0.0))
    avg_recall = max(0.0, min(1.0, avg_recall))
    avg_irrelevant = max(0.0, min(1.0, avg_irrelevant))

    # Build the report string
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    md = []
    md.append(f"# Data Insight Agent Evaluation Report ({provider.upper()})")
    md.append(f"Generated at: {now_str}")
    md.append(f"Provider: {provider}")
    if provider == "ollama":
        md.append(f"Model: {settings.OLLAMA_MODEL}")
        md.append("Base URL: local")
    md.append(f"Repetitions per case: {repetitions}")
    md.append("")

    if provider == "mock":
        md.append("> [!WARNING]")
        md.append("> **Evaluation harness verification only**")
        md.append("> This report was generated using the offline `MockProvider` and does not represent actual LLM performance.")
        md.append("")

    md.append("## Summary Statistics")
    md.append("| Metric | Formula / Source | Score | Portfolio Target | Status |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")
    md.append(f"| **Cases Attempted** | Total cases run | {total} | - | - |")
    md.append(f"| **Cases Completed** | Runs without runtime exceptions | {completed} | - | - |")
    md.append(f"| **Planner Schema-Valid Rate** | `schema_valid_runs / attempted` | {schema_valid_rate:.1%} | 100% | {'PASS' if schema_valid_rate >= 1.0 else 'FAIL'} |")
    md.append(f"| **Initial Plan-Valid Rate** | `initial_valid_runs / attempted` | {initial_plan_valid_rate:.1%} | - | - |")
    md.append(f"| **Post-Repair Plan-Valid Rate** | `validator_accepted_runs / attempted` | {post_repair_plan_valid_rate:.1%} | >= 90% | {'PASS' if post_repair_plan_valid_rate >= 0.90 else 'FAIL'} |")
    md.append(f"| **Plan Repair Rate** | `plan_repair_attempted / attempted` | {plan_repair_rate:.1%} | - | - |")
    md.append(f"| **Repair Success Rate** | `plan_repair_succeeded / plan_repair_attempted` | {repair_success_rate:.1%} | - | - |")
    md.append(f"| **Required-Operation Recall** | `selected_required_ops / expected_ops` | {avg_recall:.1%} | >= 85% | {'PASS' if avg_recall >= 0.85 else 'FAIL'} |")
    md.append(f"| **Average Irrelevant-Operation Rate** | `irrelevant_steps / total_steps` | {avg_irrelevant:.1%} | - | - |")
    md.append(f"| **Execution Correctness Rate** | `pandas_execution_correct / attempted` | {execution_rate:.1%} | 100% | {'PASS' if execution_rate >= 1.0 else 'FAIL'} |")
    md.append(f"| **Structural Grounding Rate** | `grounded_runs / completed` | {grounding_rate:.1%} | 100% | {'PASS' if grounding_rate >= 1.0 else 'FAIL'} |")
    md.append(f"| **Causal-Claim Flag Count** | Keyword count in descriptive cases | {total_causal_flags} | 0 | {'PASS' if total_causal_flags == 0 else 'FAIL'} |")
    md.append(f"| **Unsupported-Numeric-Claim Flag Count** | Metric mismatch flags in report | {total_numeric_flags} | 0 | {'PASS' if total_numeric_flags == 0 else 'FAIL'} |")
    md.append(f"| **End-to-End Success Rate** | `success_runs / attempted` | {end_to_end_rate:.1%} | >= 80% | {'PASS' if end_to_end_rate >= 0.80 else 'FAIL'} |")
    md.append(f"| **Average Latency** | Time per pipeline run | {avg_latency:.0f} ms | - | - |")

    if repetitions > 1:
        md.append(f"| **Worst-Case End-to-End Success** | Cases succeeding in all trials | {worst_case_success_rate:.1%} | - | - |")
        md.append(f"| **Provider Failure Count** | SDK / rate-limit failures | {provider_failures} | - | - |")

    # Check if any results contain human scores
    human_rated_results = [r for r in results if r.human_scores is not None]
    if human_rated_results:
        # Calculate averages
        avg_rel = sum(r.human_scores.relevance for r in human_rated_results) / len(human_rated_results)
        avg_fq = sum(r.human_scores.finding_quality for r in human_rated_results) / len(human_rated_results)
        avg_rec = sum(r.human_scores.recommendation_usefulness for r in human_rated_results) / len(human_rated_results)
        avg_rest = sum(r.human_scores.restraint for r in human_rated_results) / len(human_rated_results)
        avg_clar = sum(r.human_scores.clarity for r in human_rated_results) / len(human_rated_results)

        md.append("## Human Review Scorecard")
        md.append("| Dimension | Average Score (1-5) | Rubric Reference |")
        md.append("| :--- | :--- | :--- |")
        md.append(f"| **Relevance** | {avg_rel:.1f} / 5.0 | 1=Unrelated, 3=Acceptable, 5=Excellent direct address |")
        md.append(f"| **Finding Quality** | {avg_fq:.1f} / 5.0 | 1=Inaccurate, 3=Factually correct, 5=Deep evidence observations |")
        md.append(f"| **Recommendation Usefulness** | {avg_rec:.1f} / 5.0 | 1=Generic/Useless, 3=Actionable but standard, 5=Highly prioritized |")
        md.append(f"| **Restraint** | {avg_rest:.1f} / 5.0 | 1=Overclaims/Causal bias, 3=Conservative, 5=Perfect bounds & limitations |")
        md.append(f"| **Clarity** | {avg_clar:.1f} / 5.0 | 1=confusing/disorganized, 3=understandable/organized, 5=concise/logical/actionable |")
        md.append("")

        md.append("## Case Breakdown")
        md.append("| Case ID | Model | Success | Latency | Recall | Irrelevant | Causal Flags | Num Flags | Repair Attempted | Repair Succeeded | Relevance | Finding | Recs | Restraint | Clarity |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for r in results:
            h = r.human_scores
            h_rel = h.relevance if h else "-"
            h_fq = h.finding_quality if h else "-"
            h_rec = h.recommendation_usefulness if h else "-"
            h_rest = h.restraint if h else "-"
            h_clar = h.clarity if h else "-"
            md.append(
                f"| `{r.case_id}` | `{r.model}` | {'✅' if r.final_success else '❌'} | "
                f"{r.latency_ms:.0f} ms | {r.planner_scores.required_operation_recall:.0%} | "
                f"{r.planner_scores.irrelevant_operation_rate:.0%} | {r.grounding_scores.causal_claim_flags} | "
                f"{r.grounding_scores.unsupported_numeric_claim_flags} | "
                f"{'Yes' if r.planner_scores.plan_repair_attempted else 'No'} | "
                f"{'Yes' if r.planner_scores.plan_repair_succeeded else 'No'} | "
                f"{h_rel} | {h_fq} | {h_rec} | {h_rest} | {h_clar} |"
            )
    else:
        md.append("## Case Breakdown")
        md.append("| Case ID | Model | Success | Latency | Error Category | Recall | Irrelevant Rate | Causal Flags | Num Flags | Repair Attempted | Repair Succeeded |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for r in results:
            err = r.error_category if r.error_category else "None"
            md.append(
                f"| `{r.case_id}` | `{r.model}` | {'✅' if r.final_success else '❌'} | "
                f"{r.latency_ms:.0f} ms | `{err}` | {r.planner_scores.required_operation_recall:.0%} | "
                f"{r.planner_scores.irrelevant_operation_rate:.0%} | {r.grounding_scores.causal_claim_flags} | "
                f"{r.grounding_scores.unsupported_numeric_claim_flags} | "
                f"{'Yes' if r.planner_scores.plan_repair_attempted else 'No'} | "
                f"{'Yes' if r.planner_scores.plan_repair_succeeded else 'No'} |"
            )

    return "\n".join(md)


def save_evaluation_artifacts(results: List[EvaluationResult], provider: str, repetitions: int, output_dir: str = "evaluation_results"):
    """Saves both raw JSON log files and readable Markdown report files."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # 1. Save JSON Log
    json_path = os.path.join(output_dir, f"{timestamp}_{provider}_results.json")
    # Convert list of models to dictionaries
    serializable_results = [r.model_dump() for r in results]
    json_data = {
        "provider": provider,
        "repetitions": repetitions,
        "timestamp": timestamp,
        "results": serializable_results
    }
    if provider == "ollama":
        json_data["model"] = settings.OLLAMA_MODEL
        json_data["base_url"] = "local"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    # 2. Save Markdown summary
    md_path = os.path.join(output_dir, f"{timestamp}_{provider}_summary.md")
    markdown_content = compile_metrics_markdown(results, provider, repetitions)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    return json_path, md_path
