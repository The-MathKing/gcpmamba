import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(22, 3.5))

boxes = [
    ('Continuous Co-expression\nMatrix $D$ ($N \\times N$)', 'lightblue'),
    ('Gating Matrix $W_g$ &\nSigmoid Activation', 'lightgreen'),
    ('Dimension Averaging\n$\\rightarrow M_\\Delta$ ($N \\times 1$)', 'yellow'),
    ('Continuous-Time\nDiscretization\n(Mamba SSM)', 'orange'),
    ('Predicted Expression\nTensors ($\\mathcal{O}(L)$ compute)', 'salmon')
]

x_centers = np.linspace(0.1, 0.9, 5)

for i, (text, color) in enumerate(boxes):
    plt.text(x_centers[i], 0.5, text, fontsize=16, ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.5", fc=color, ec="gray", lw=2))
    
    # Draw arrow to the next box
    if i < len(boxes) - 1:
        # Arrow starts right after current text box and ends just before next text box
        plt.annotate('', xy=(x_centers[i+1] - 0.08, 0.5), xytext=(x_centers[i] + 0.08, 0.5),
                     arrowprops=dict(arrowstyle="->", lw=4, color='black'))

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')
plt.tight_layout()
plt.savefig('architecture_diagram.png', dpi=300, bbox_inches='tight')
