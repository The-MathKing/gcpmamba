import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.text(0.5, 0.9, 'Topological Graph Matrix $D$ ($N \\times N$)', fontsize=14, ha='center', bbox=dict(boxstyle="round,pad=0.5", fc="lightblue"))
plt.arrow(0.5, 0.85, 0, -0.1, head_width=0.02, head_length=0.02, fc='k', ec='k')
plt.text(0.5, 0.7, 'Gating Matrix $W_g$ & Sigmoid Activation', fontsize=14, ha='center', bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen"))
plt.arrow(0.5, 0.65, 0, -0.1, head_width=0.02, head_length=0.02, fc='k', ec='k')
plt.text(0.5, 0.5, 'Dimension Averaging $\\rightarrow M_\\Delta$ ($N \\times 1$)', fontsize=14, ha='center', bbox=dict(boxstyle="round,pad=0.5", fc="yellow"))
plt.arrow(0.5, 0.45, 0, -0.1, head_width=0.02, head_length=0.02, fc='k', ec='k')
plt.text(0.5, 0.3, 'Continuous-Time Discretization (Mamba SSM Block)', fontsize=14, ha='center', bbox=dict(boxstyle="round,pad=0.5", fc="orange"))
plt.arrow(0.5, 0.25, 0, -0.1, head_width=0.02, head_length=0.02, fc='k', ec='k')
plt.text(0.5, 0.1, 'Predicted Expression Tensors ($\\mathcal{O}(N)$ compute)', fontsize=14, ha='center', bbox=dict(boxstyle="round,pad=0.5", fc="salmon"))

plt.axis('off')
plt.tight_layout()
plt.savefig('architecture_diagram.png', dpi=300)
