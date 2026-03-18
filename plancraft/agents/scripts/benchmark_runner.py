
import subprocess
import time
import argparse
import os
import json
from datetime import datetime

# Architectures to benchmark
ARCHITECTURES = ["single", "independent", "centralized", "decentralized", "hybrid"]

BACKEND_SCRIPTS = {
    "copilot": "eval_copilot.py",
}

def analyze_results(json_data):
    """Parses raw JSON results into complexity and failure stats."""
    results = json_data["results"]
    
    # Complexity Analysis
    complexities = {"Low (1-2)": [], "Medium (3-8)": [], "High (9+)": []}
    failures = {}
    
    for r in results:
        comp = r.get("complexity")
        if comp is None:
            comp = 0  # Treat None as 0 for safe comparison, though these are typically impossible tasks
            
        category = "High (9+)"
        if comp <= 2:
            category = "Low (1-2)"
        elif comp <= 8:
            category = "Medium (3-8)"
            
        complexities[category].append(r)
        
        if not r["success"]:
            reason = r.get("reason", "unknown")
            failures[reason] = failures.get(reason, 0) + 1
            
    comp_stats = {}
    for cat, items in complexities.items():
        if not items:
            comp_stats[cat] = {"total": 0, "success": 0, "rate": 0, "steps": 0}
            continue
            
        successes = [i for i in items if i["success"]]
        rate = len(successes) / len(items) if items else 0
        avg_steps = sum(i["steps"] for i in successes) / len(successes) if successes else 0
        
        comp_stats[cat] = {
            "total": len(items),
            "success": len(successes),
            "rate": rate,
            "steps": avg_steps
        }
        
    return comp_stats, failures

def generate_markdown_report(report_data, model, split, date_str):
    """Generates a markdown report string."""
    lines = []
    lines.append(f"# Research Benchmark: Copilot Evaluation Report")
    lines.append(f"**Model**: {model}")
    lines.append(f"**Dataset**: Plancraft ({split})")
    lines.append(f"**Date**: {date_str}")
    lines.append("")
    
    lines.append("## 1. Executive Summary")
    lines.append("| Architecture | Success Rate | Avg Steps |")
    lines.append("|--------------|--------------|-----------|")
    
    for arch in ARCHITECTURES:
        if arch in report_data and isinstance(report_data[arch], dict):
            status = report_data[arch]
            rate = status.get("success_rate", 0)
            steps = status.get("avg_steps", 0)
            lines.append(f"| {arch.capitalize()} | {rate:.1%} | {steps:.1f} |")
        else:
            lines.append(f"| {arch.capitalize()} | N/A | N/A |")
            
    lines.append("")
    
    for arch in ARCHITECTURES:
        if arch not in report_data or not isinstance(report_data[arch], dict):
            continue
            
        data = report_data[arch]
        comp_stats = data.get("complexity_stats", {})
        failures = data.get("failures", {})
        
        lines.append(f"## {arch.capitalize()} Analysis")
        lines.append("### Complexity Breakdown")
        lines.append("| Complexity | Total | Success | Rate | Avg Steps |")
        lines.append("|------------|-------|---------|------|-----------|")
        
        # Order: Low, Medium, High
        for cat in ["Low (1-2)", "Medium (3-8)", "High (9+)"]:
            stats = comp_stats.get(cat, {})
            if stats:
                lines.append(f"| {cat} | {stats['total']} | {stats['success']} | {stats['rate']:.1%} | {stats['steps']:.1f} |")
                
        lines.append("")
        lines.append("### Top Failures")
        if failures:
            sorted_fails = sorted(failures.items(), key=lambda x: x[1], reverse=True)
            for reason, count in sorted_fails[:5]:
                lines.append(f"- `{reason}`: {count}")
        else:
            lines.append("- None")
        lines.append("")
        
    return "\n".join(lines)


def run_benchmark(limit: int, model: str, max_steps: int, output_dir: str, backend: str, split: str):
    """
    Runs the benchmark to completion and saves a markdown report.
    """
    os.makedirs(output_dir, exist_ok=True)
    report = {}
    
    eval_script = "eval_copilot.py"
    print(f"🔧 Backend: copilot ({eval_script})")
    print(f"🤖 Model: {model}")
    print(f"📂 Split: {split} (Limit: {limit})")
    
    
    target_dir = "plancraft/agents/benchmark/output"
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs("plancraft/agents/benchmark/logs", exist_ok=True)
    
    processes = []
    
    for arch in ARCHITECTURES:
        prefix = "copilot"
        existing_files = [f for f in os.listdir(target_dir) if f.endswith(".json") and arch in f and prefix in f and split in f]
        if existing_files:
            print(f"⏩ Skipping {arch} (Already completed)")
            # Load existing data for report
            last_file = os.path.join(target_dir, sorted(existing_files, key=lambda x: os.path.getmtime(os.path.join(target_dir, x)), reverse=True)[0])
            with open(last_file) as f:
                raw_data = json.load(f)
            
            comp_stats, failures = analyze_results(raw_data)
            report[arch] = {
                "success_rate": raw_data["success_rate"],
                "avg_steps": sum(r["steps"] for r in raw_data["results"]) / len(raw_data["results"]) if raw_data.get("results") else 0,
                "complexity_stats": comp_stats,
                "failures": failures,
                "file": last_file
            }
            continue

        print(f"\n🚀 eg benchmark for: {arch}")
        
        cmd = [
            "python3", "-u", eval_script,
            "--split", split,
            "--max-examples", str(limit),
            "--max-steps", str(max_steps),
            "--model", model,
            "--architecture", arch
        ]
        
        log_file = open(f"plancraft/agents/benchmark/logs/{arch}.log", "w")
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((arch, p, log_file))
        time.sleep(5)  # Stagger client initializations

    for arch, p, log_file in processes:
        p.wait()
        log_file.close()
        
        if p.returncode == 0:
            output_files = sorted(
                [f for f in os.listdir(target_dir) if f.endswith(".json") and arch in f and prefix in f],
                key=lambda x: os.path.getmtime(os.path.join(target_dir, x)),
                reverse=True
            )
            
            if output_files:
                last_file = os.path.join(target_dir, output_files[0])
                with open(last_file) as f:
                    raw_data = json.load(f)
                
                comp_stats, failures = analyze_results(raw_data)
                
                report[arch] = {
                    "success_rate": raw_data["success_rate"],
                    "avg_steps": sum(r["steps"] for r in raw_data["results"]) / len(raw_data["results"]) if raw_data.get("results") else 0,
                    "complexity_stats": comp_stats,
                    "failures": failures,
                    "file": last_file
                }
                print(f"✅ {arch} done. Success: {raw_data['success_rate']:.1%}")
            else:
                print(f"⚠ Output file missing for {arch}")
        else:
            print(f"❌ {arch} failed with return code {p.returncode}. See logs/{arch}.log")

    # Generate Markdown Report
    date_str = datetime.now().strftime('%Y-%m-%d')
    md_content = generate_markdown_report(report, model, split, date_str)
    
    md_file = "plancraft/agents/benchmark/BENCHMARK_REPORT.md"
    with open(md_file, "w") as f:
        f.write(md_content)
        
    print(f"\n🏁 Benchmark complete! Report saved to {md_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=110, help="Examples per agent (0=all)")
    parser.add_argument("--model", default="gpt-5-mini", help="Model to use")
    parser.add_argument("--max-steps", type=int, default=30, help="Max steps per example")
    parser.add_argument("--output-dir", default="benchmark_results", help="Directory for summary")
    parser.add_argument("--split", default="val.small", help="Dataset split (e.g. val.small)")
    parser.add_argument("--backend", default="copilot", choices=["gemini", "copilot"])
    
    args = parser.parse_args()
    
    run_benchmark(args.limit, args.model, args.max_steps, args.output_dir, args.backend, args.split)
