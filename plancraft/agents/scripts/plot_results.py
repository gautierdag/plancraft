
import json
import os
import matplotlib.pyplot as plt
import argparse
import glob

def plot_results(summary_file: str, output_dir: str):
    with open(summary_file) as f:
        data = json.load(f)
        
    archs = list(data.keys())
    success_rates = [data[arch]["success_rate"] * 100 for arch in archs if isinstance(data[arch], dict)]
    avg_steps = [data[arch]["avg_steps"] for arch in archs if isinstance(data[arch], dict)]
    valid_archs = [arch for arch in archs if isinstance(data[arch], dict)]
    
    # Plot Success Rate
    plt.figure(figsize=(10, 6))
    plt.bar(valid_archs, success_rates, color=['blue', 'green', 'orange', 'red', 'purple'])
    plt.title("Success Rate by Architecture")
    plt.ylabel("Success Rate (%)")
    plt.ylim(0, 100)
    for i, v in enumerate(success_rates):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    plt.savefig(os.path.join(output_dir, "success_rate.png"))
    
    # Plot Average Steps
    plt.figure(figsize=(10, 6))
    plt.bar(valid_archs, avg_steps, color=['blue', 'green', 'orange', 'red', 'purple'])
    plt.title("Average Steps by Architecture")
    plt.ylabel("Average Steps")
    for i, v in enumerate(avg_steps):
        plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
    plt.savefig(os.path.join(output_dir, "avg_steps.png"))
    
    print(f"plots saved to {output_dir}")

def generate_markdown_summary(summary_file: str):
    with open(summary_file) as f:
        data = json.load(f)
        
    print("\n## Benchmark Summary\n")
    print("| Architecture | Success Rate | Avg Steps |")
    print("|--------------|--------------|-----------|")
    for arch, metrics in data.items():
        if isinstance(metrics, dict):
             print(f"| {arch} | {metrics['success_rate']:.1%} | {metrics['avg_steps']:.1f} |")
        else:
             print(f"| {arch} | Failed | - |")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-file", help="Path to summary JSON")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output dir")
    args = parser.parse_args()
    
    if not args.summary_file:
        # Find latest summary file
        files = glob.glob(os.path.join(args.output_dir, "benchmark_summary_*.json"))
        if files:
            args.summary_file = max(files, key=os.path.getctime)
        else:
            print("No summary file found.")
            exit(1)
            
    plot_results(args.summary_file, args.output_dir)
    generate_markdown_summary(args.summary_file)
