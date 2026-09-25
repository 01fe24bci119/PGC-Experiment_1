import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_performance_graph():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    csv_path = os.path.join(repo_root, "results", "execution_times.csv")
    output_png_path = os.path.join(repo_root, "results", "performance_comparison.png")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    # Read data from CSV (no hardcoding)
    df = pd.read_csv(csv_path)

    # Style configuration
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    implementations = df["Implementation"]
    execution_times = df["Execution_Time_Seconds"]
    threads = df["Threads"]
    speedups = df["Speedup"]

    colors = ["#2563EB", "#10B981"]  # Sequential: Royal Blue, OpenMP: Emerald Green
    bars = ax.bar(implementations, execution_times, color=colors, width=0.45, edgecolor="#1E293B", linewidth=1.2)

    # Annotate bars with execution time and metadata
    for bar, time_val, th, sp in zip(bars, execution_times, threads, speedups):
        y_pos = bar.get_height()
        label = f"{time_val:.2f} s\n({th} Thread{'s' if th > 1 else ''})"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos + 4,
            label,
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold',
            color='#1E293B'
        )

    # Calculate speedup from the dataset
    seq_time = df.loc[df["Implementation"] == "Sequential", "Execution_Time_Seconds"].values[0]
    omp_time = df.loc[df["Implementation"] == "OpenMP", "Execution_Time_Seconds"].values[0]
    calc_speedup = seq_time / omp_time

    # Add speedup callout box
    ax.text(
        0.5,
        max(execution_times) * 0.72,
        f"Speedup: {calc_speedup:.2f}× faster\nEfficiency: {(calc_speedup / 8) * 100:.2f}% (8 threads)",
        ha='center',
        va='center',
        fontsize=12,
        fontweight='semibold',
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#FEF3C7", edgecolor="#F59E0B", linewidth=1.5)
    )

    # Chart details
    ax.set_ylabel("Execution Time (seconds)", fontsize=13, fontweight='bold', labelpad=10)
    ax.set_xlabel("Implementation", fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title("Matrix Multiplication Performance: Sequential vs OpenMP\n(Matrix Size: 4000 × 4000)", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, max(execution_times) * 1.25)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Source watermark
    plt.figtext(0.95, 0.02, "Measured on Ubuntu WSL2 (GCC 15.2.0 -O2)", ha='right', fontsize=9, color="#64748B", fontstyle='italic')

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300)
    plt.close()
    print(f"Graph successfully generated and saved to: {output_png_path}")

if __name__ == "__main__":
    generate_performance_graph()
