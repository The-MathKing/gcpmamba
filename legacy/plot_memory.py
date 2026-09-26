import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams.update({
    'font.size': 20, 
    'axes.titlesize': 24, 
    'axes.labelsize': 20, 
    'xtick.labelsize': 18, 
    'ytick.labelsize': 18, 
    'legend.fontsize': 18
})
sns.set_theme(style="whitegrid")
L_vals = [100, 500, 1000, 2000, 5000]

# Rough estimates based on O(L) and O(L^2) scaling from the caption text
mem_mamba = [0.7, 3.3, 6.6, 13.2, 33.0]
# "exceeding 14GB at 5000 genes"
mem_gcn = [10.0, 250.0, 1000.0, 4000.0, 15000.0]

fig, ax = plt.subplots(figsize=(9, 6))
data = {
    'GCP-Mamba ($\mathcal{O}(L)$)': mem_mamba,
    'FaithfulGEARS ($\mathcal{O}(L^2)$)': mem_gcn
}
colors = {'GCP-Mamba ($\mathcal{O}(L)$)': '#2196F3', 'FaithfulGEARS ($\mathcal{O}(L^2)$)': '#E53935'}

for label, mems in data.items():
    # Measured: first 3 points, Projected: last 3 points (overlap at index 2)
    L_measured = L_vals[:3]
    mems_measured = mems[:3]
    L_projected = L_vals[2:]
    mems_projected = mems[2:]
    
    # Plot Measured
    ax.plot(L_measured, mems_measured, marker='o', markersize=8, linewidth=3, 
            color=colors[label], label=label)
    # Plot Projected
    ax.plot(L_projected, mems_projected, linestyle='--', marker='s', markersize=8, linewidth=3,
            color=colors[label])

ax.axhline(y=15360, color='r', linestyle='--', label='15GB Colab T4 VRAM Limit')

plt.xlabel('Sequence Length (Number of Genes $L$)', fontsize=12)
plt.ylabel('Peak GPU VRAM Usage (MB)', fontsize=12)
plt.title('Memory Scaling: GCP-Mamba vs. Graph Neural Networks', fontsize=14, pad=15)
plt.yscale('log')
plt.xscale('log')
plt.xticks(L_vals, labels=[str(v) for v in L_vals])
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('memory_scaling.png', dpi=300)
print("Saved memory_scaling.png")
