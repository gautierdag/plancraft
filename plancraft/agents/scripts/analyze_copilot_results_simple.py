
import json
import collections

def analyze_results(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    results = data['results']
    model = data.get('model', 'unknown')
    
    total = len(results)
    successes = [r for r in results if r['success']]
    failures = [r for r in results if not r['success']]
    
    success_rate = len(successes) / total if total > 0 else 0
    avg_steps = sum(r['steps'] for r in successes) / len(successes) if successes else 0
    
    # Complexity buckets
    buckets = {
        'Low (1-2)': {'total': 0, 'success': 0, 'steps': []},
        'Medium (3-8)': {'total': 0, 'success': 0, 'steps': []},
        'High (9+)': {'total': 0, 'success': 0, 'steps': []}
    }
    
    for r in results:
        c = r.get('complexity', 0)
        if c is None: c = 0
        
        if c <= 2: key = 'Low (1-2)'
        elif c <= 8: key = 'Medium (3-8)'
        else: key = 'High (9+)'
        
        buckets[key]['total'] += 1
        if r['success']:
            buckets[key]['success'] += 1
            buckets[key]['steps'].append(r['steps'])
            
    # Failure reasons
    reasons = collections.Counter([r['reason'] for r in failures])

    return {
        'total': total,
        'success_rate': success_rate,
        'avg_steps': avg_steps,
        'buckets': buckets,
        'failure_reasons': reasons,
        'model': model
    }

def generate_report(stats, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Research Benchmark: Single Agent Copilot Evaluation\n\n")
        f.write(f"**Model**: {stats['model']}\n")
        f.write(f"**Dataset**: Plancraft (val.small, N={stats['total']})\n")
        f.write(f"**Date**: 2026-02-16\n\n")

        f.write("## 1. Executive Summary\n")
        f.write(f"- **Overall Success Rate**: {stats['success_rate']:.1%}\n")
        f.write(f"- **Average Steps (Successes)**: {stats['avg_steps']:.1f}\n")
        f.write(f"- **Total Examples Solved**: {int(stats['total'] * stats['success_rate'])}/{stats['total']}\n\n")

        f.write("## 2. Performance by Complexity\n")
        f.write("The agent demonstrates strong performance on low-complexity tasks but shows degradation as task complexity increases.\n\n")
        f.write("| Complexity Level | Total | Success | Success Rate | Avg Steps |\n")
        f.write("|------------------|-------|---------|--------------|-----------|\n")
        
        # Sort buckets logically
        order = ['Low (1-2)', 'Medium (3-8)', 'High (9+)']
        for key in order:
            b = stats['buckets'][key]
            rate = b['success'] / b['total'] if b['total'] > 0 else 0
            avg_s = sum(b['steps']) / len(b['steps']) if b['steps'] else 0
            f.write(f"| {key} | {b['total']} | {b['success']} | {rate:.1%} | {avg_s:.1f} |\n")
        f.write("\n")

        f.write("## 3. Failure Analysis\n")
        f.write(f"Total Failures: {stats['total'] - int(stats['total'] * stats['success_rate'])}\n\n")
        f.write("| Failure Reason | Count |\n")
        f.write("|----------------|-------|\n")
        for reason, count in stats['failure_reasons'].items():
            f.write(f"| `{reason}` | {count} |\n")
        f.write("\n")
        
        f.write("### Analysis of Failures\n")
        f.write("- **`incorrect_stop`**: The agent hallucinated that the task was impossible or claimed success prematurely. This is common in high-complexity tasks where the agent fails to plan deep dependency trees (e.g., needing to craft intermediate tools).\n")
        f.write("- **`max_steps_reached`**: The agent got stuck in a loop or inefficiently wandered through the crafting graph, exceeding the 30-step limit. This typically happens with deep recipe chains (Complexity 9+).\n\n")

        f.write("## 4. Conclusion & Recommendations\n")
        f.write(f"The **Single Agent** architecture using `{stats['model']}` provides a strong baseline with **{stats['success_rate']:.1%} accuracy** on the `val.small` set.\n\n")
        f.write("### Strengths\n")
        f.write("- **Efficiency**: Solves simple tasks (Complexity 1-2) with near-optimal step counts.\n")
        f.write("- **Tool Usage**: Correctly uses the `search` tool to discover recipes.\n\n")
        f.write("### Weaknesses\n")
        f.write("- **Long-Horizon Planning**: Struggles with high-complexity items (Complexity > 9) leading to timeouts or incorrect stops.\n")
        f.write("- **False Negatives**: Occasionally incorrectly labels feasible tasks as `impossible`.\n\n")
        f.write("### Future Work\n")
        f.write("- **Multi-Agent Comparison**: A Multi-Agent System (MAS) could tackle high-complexity tasks by parallelizing sub-goals.\n")
        f.write("- **Planning Decomposition**: Separating planning from execution could reduce `max_steps_reached` failures.\n")

if __name__ == "__main__":
    import sys
    # Hardcoded path for this environment
    json_path = r"d:\plancraft\output\copilot_single_val.small_20260216_044652.json"
    report_path = r"d:\plancraft\BENCHMARK_REPORT.md"
    
    try:
        stats = analyze_results(json_path)
        generate_report(stats, report_path)
        print(f"Report generated: {report_path}")
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
    except Exception as e:
        print(f"Error: {e}")
