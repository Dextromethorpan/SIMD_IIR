import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV produced by the C++ benchmark
df = pd.read_csv("build/Release/bench_results_cache.csv")

# Median across trials
med = df.groupby(["version","N_taps","S"], as_index=False)["ns_per_sample_per_tap"].median()

# One figure per N (number of taps)
for N in sorted(med["N_taps"].unique()):
    plt.figure(figsize=(7,4))
    subset = med[med["N_taps"]==N]
    for v in ["scalar","sse","avx"]:
        d = subset[subset["version"]==v]
        plt.semilogx(d["S"], d["ns_per_sample_per_tap"], marker="o", label=v.upper())
    plt.xlabel("Number of samples (log scale)")
    plt.ylabel("ns per sample per tap")
    plt.title(f"IIR Filter — Cache Dependency, N={N} taps")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"bench_cache_N{N}.png")

print("Saved cache dependency plots: bench_cache_N2.png, bench_cache_N10.png")
