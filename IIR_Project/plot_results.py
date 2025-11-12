import sys
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python plot_results.py bench_results.csv")
    exit(1)

df = pd.read_csv(sys.argv[1])
med = df.groupby(["version","N_taps"], as_index=False)["ns_per_sample_per_tap"].median()

plt.figure(figsize=(7,4.5))
for v in ["scalar","sse","avx"]:
    d = med[med["version"]==v]
    plt.plot(d["N_taps"], d["ns_per_sample_per_tap"], marker="o", label=v.upper())
plt.xlabel("Number of IIR taps (N)")
plt.ylabel("ns per sample per tap")
plt.title("IIR Filter — Scalar vs SSE vs AVX (fused-tap)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("bench_plot.png")
print("Saved bench_plot.png")
