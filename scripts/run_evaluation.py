import os
import sys
import argparse
import asyncio
import json

# Setup workspace PYTHONPATH routing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.runner import load_cases, evaluate_case
from evaluation.report import save_evaluation_artifacts, compile_metrics_markdown
from evaluation.models import EvaluationResult, HumanScores
from app.config import settings

def prompt_human_score(dimension: str, rubric: str) -> int:
    """Helper to prompt for integer human score between 1 and 5."""
    while True:
        val = input(f"      {dimension} (1-5) [{rubric}]: ").strip()
        if val in ["1", "2", "3", "4", "5"]:
            return int(val)
        print("        Invalid input. Must be an integer between 1 and 5.")

async def main():
    parser = argparse.ArgumentParser(description="Data Insight Agent Local Evaluation Harness")
    parser.add_argument(
        "--provider",
        choices=["mock", "ollama", "gemini", "openai", "all"],
        default="mock",
        help="Select LLM provider to evaluate. 'all' evaluates active credentials."
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of times to run each case."
    )
    parser.add_argument(
        "--cases-dir",
        default="evaluation/cases",
        help="Directory containing JSON evaluation cases."
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results",
        help="Directory to save output MD reports and JSON metrics logs."
    )
    parser.add_argument(
        "--review",
        help="Path to a JSON evaluation results log file to annotate with human review scores."
    )

    args = parser.parse_args()

    # Check if we are running in review mode
    if args.review:
        if not os.path.exists(args.review):
            print(f"Error: Review JSON file not found at: {args.review}")
            sys.exit(1)

        print(f"=== Starting Interactive Human Review for Log: {args.review} ===")
        try:
            with open(args.review, "r", encoding="utf-8") as f:
                data = json.load(f)
            provider = data.get("provider", "mock")
            repetitions = data.get("repetitions", 1)
            timestamp = data.get("timestamp", "reviewed")
            results = [EvaluationResult.model_validate(r) for r in data.get("results", [])]
        except Exception as e:
            print(f"Error loading or parsing JSON log file: {str(e)}")
            sys.exit(1)

        # Load cases for questions reference lookup
        try:
            cases = load_cases(args.cases_dir)
            case_map = {c.case_id: c for c in cases}
        except Exception as e:
            print(f"Warning: Could not load cases directory for references: {str(e)}")
            case_map = {}

        # Interactive review loop
        print("\nEnter human review scores (1-5) for each case result.")
        for idx, res in enumerate(results):
            case = case_map.get(res.case_id)
            question = case.question if case else "Unknown Question"

            print(f"\n[{idx + 1}/{len(results)}] Reviewing Case '{res.case_id}' (Model: {res.model})")
            print(f"    Question: {question}")
            print(f"    Error:    {res.error_category or 'None'}")
            print(f"    Success:  {res.final_success}")

            rel = prompt_human_score("Relevance", "1=Unrelated, 3=Acceptable, 5=Excellent direct address")
            fq = prompt_human_score("Finding Quality", "1=Inaccurate, 3=Basic, 5=Deep evidence observations")
            rec = prompt_human_score("Recommendation Usefulness", "1=Generic, 3=Actionable, 5=Highly prioritized")
            rest = prompt_human_score("Restraint", "1=Overclaims/Causal, 3=Conservative, 5=Perfect bounds")
            clar = prompt_human_score("Clarity", "1=confusing/disorganized, 3=understandable/organized, 5=concise/logical/actionable")

            res.human_scores = HumanScores(
                relevance=rel,
                finding_quality=fq,
                recommendation_usefulness=rec,
                restraint=rest,
                clarity=clar
            )

        # Save updated JSON log back to original location
        try:
            with open(args.review, "w", encoding="utf-8") as f:
                json.dump({
                    "provider": provider,
                    "repetitions": repetitions,
                    "timestamp": timestamp,
                    "results": [r.model_dump() for r in results]
                }, f, indent=2)

            # Overwrite the companion Markdown summary report
            md_filepath = args.review.replace("_results.json", "_summary.md")
            markdown_content = compile_metrics_markdown(results, provider, repetitions)
            with open(md_filepath, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            print(f"\nReview successfully saved!")
            print(f"  Updated JSON log: {args.review}")
            print(f"  Updated Markdown: {md_filepath}")
        except Exception as e:
            print(f"Error saving reviewed artifacts: {str(e)}")
            sys.exit(1)

        sys.exit(0)

    providers_to_run = []
    if args.provider == "all":
        providers_to_run.append("mock")
        # Only add real providers if credentials exist
        if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY.strip():
            providers_to_run.append("gemini")
        else:
            print("Skipping gemini provider evaluation: GEMINI_API_KEY is not configured.")

        if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.strip():
            providers_to_run.append("openai")
        else:
            print("Skipping openai provider evaluation: OPENAI_API_KEY is not configured.")
    else:
        # Check credentials if a specific real provider was requested
        prov = args.provider
        if prov == "gemini":
            if not settings.GEMINI_API_KEY or not settings.GEMINI_API_KEY.strip():
                print("Error: GEMINI_API_KEY is not configured. Cannot evaluate gemini.")
                sys.exit(1)
        elif prov == "openai":
            if not settings.OPENAI_API_KEY or not settings.OPENAI_API_KEY.strip():
                print("Error: OPENAI_API_KEY is not configured. Cannot evaluate openai.")
                sys.exit(1)
        providers_to_run.append(prov)

    # Execute runs
    for provider in providers_to_run:
        if provider == "ollama":
            import httpx
            url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags"
            try:
                response = httpx.get(url, timeout=3.0)
                if response.status_code == 200:
                    models_data = response.json()
                    pulled_models = [m["name"] for m in models_data.get("models", [])]
                    target_model = settings.OLLAMA_MODEL
                    found = False
                    for m in pulled_models:
                        if m == target_model or m.split(":")[0] == target_model.split(":")[0]:
                            found = True
                            if m != target_model:
                                print(f"Warning: Exact model tag '{target_model}' not found, but a match '{m}' is available. Using '{m}' instead.")
                                settings.OLLAMA_MODEL = m
                            break
                    if not found:
                        print(f"Error: Model '{target_model}' is not pulled in local Ollama instance.")
                        print(f"Please run: ollama pull {target_model}")
                        sys.exit(1)
                else:
                    print(f"Error: Local Ollama returned status {response.status_code} at {url}.")
                    sys.exit(1)
            except Exception as e:
                print(f"Error: Local Ollama service is not running or unreachable at {settings.OLLAMA_BASE_URL}.")
                print("Start the Ollama application and ensure the server is listening.")
                sys.exit(1)

        print(f"\n=== Running Evaluation for Provider: '{provider}' ===")
        print(f"Loading cases from '{args.cases_dir}'...")
        try:
            cases = load_cases(args.cases_dir)
        except Exception as e:
            print(f"Error loading evaluation cases: {str(e)}")
            sys.exit(1)

        print(f"Successfully loaded {len(cases)} cases.")
        results = []
        for case in cases:
            print(f"  Running case '{case.case_id}' (x{args.repetitions} repetitions)...")
            for rep in range(args.repetitions):
                try:
                    result = await evaluate_case(case, provider)
                    results.append(result)
                    status_str = "[PASS]" if result.final_success else "[FAIL]"
                    err_info = f" (Error: {result.error_category})" if result.error_category else ""
                    print(f"    Repetition {rep + 1}/{args.repetitions}: {status_str}{err_info}")
                except Exception as e:
                    print(f"    Repetition {rep + 1}/{args.repetitions}: [CRASH] System Failure: {str(e)}")

        # Compile and save output artifacts
        json_path, md_path = save_evaluation_artifacts(results, provider, args.repetitions, args.output_dir)
        print(f"Evaluation for '{provider}' completed!")
        print(f"  JSON raw metrics: {json_path}")
        print(f"  Markdown report:  {md_path}")

if __name__ == "__main__":
    asyncio.run(main())
