#!/usr/bin/env python3
"""
CLI runner for Mortgage Exception Workbench Evaluation Suite.
Usage:
    PYTHONPATH=. python scripts/run_evals.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evals import (
    run_evaluation_suite,
    generate_markdown_report,
    export_json_report,
)

BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def main() -> None:
    print(f"\n{BOLD}{CYAN}======================================================================{RESET}")
    print(f"{BOLD}{CYAN}   MORTGAGE EXCEPTION WORKBENCH — EVALUATION & BENCHMARK SUITE       {RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}\n")

    start_time = time.time()
    print("⏳ Executing evaluation suite across all benchmark categories...")
    eval_summary = run_evaluation_suite()
    elapsed = time.time() - start_time

    metrics = eval_summary["metrics"]
    sla = eval_summary["sla_checks"]
    total = eval_summary["total_test_cases"]
    passed = eval_summary["total_passed"]
    failed = eval_summary["total_failed"]
    all_met = eval_summary["all_slas_met"]

    print(f"\n{BOLD}Evaluation completed in {elapsed:.2f}s{RESET}")
    print(f"Summary: {GREEN if failed == 0 else RED}{passed}/{total} Passed ({metrics['overall_pass_rate']:.1%}){RESET}\n")

    print(f"{BOLD}--- SLA Performance Benchmarks ---{RESET}")
    print(f"{'Metric':<40} {'Score':<10} {'Target':<10} {'Status':<10}")
    print("-" * 70)

    metric_labels = {
        "retrieval_hit_rate_at_1": "Retrieval Hit Rate @ 1",
        "retrieval_hit_rate_at_3": "Retrieval Hit Rate @ 3",
        "mean_reciprocal_rank": "Mean Reciprocal Rank (MRR)",
        "action_routing_accuracy": "Action Routing Accuracy",
        "pii_detection_recall": "PII Detection & Redaction",
        "injection_defense_rate": "Prompt Injection Defense",
        "policy_enforcement_rate": "Output Policy Enforcement",
        "citation_fidelity_rate": "Citation Fidelity / Hallucination",
    }

    for key, label in metric_labels.items():
        if key in sla:
            info = sla[key]
            status_str = f"{GREEN}PASS{RESET}" if info["met"] else f"{RED}FAIL{RESET}"
            print(f"{label:<40} {info['score']:<10.1%} >={info['target']:<8.1%} {status_str}")

    print("-" * 70)

    # Export reports
    report_json_path = PROJECT_ROOT / "data" / "eval_report.json"
    report_md_path = PROJECT_ROOT / "eval_report.md"

    export_json_report(eval_summary, report_json_path)
    report_md = generate_markdown_report(eval_summary)
    report_md_path.write_text(report_md, encoding="utf-8")

    print(f"\n📄 Saved JSON report to: {CYAN}{report_json_path}{RESET}")
    print(f"📄 Saved Markdown report to: {CYAN}{report_md_path}{RESET}")

    if all_met and failed == 0:
        print(f"\n{GREEN}{BOLD}🎉 ALL EVALUATIONS AND BENCHMARK SLAS PASSED!{RESET}\n")
        sys.exit(0)
    else:
        print(f"\n{RED}{BOLD}⚠️ SOME EVALUATIONS FAILED OR DID NOT MEET SLAS.{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
