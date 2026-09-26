import json
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

sns.set_theme(style="whitegrid", font_scale=1.1)

with open("training_losses.json", "r") as f:
    loss_history = json.load(f)

# loss_history[seed][model] = [loss_epoch1, ..., loss_epoch50]
epochs = len(list(loss_history.values())[0]["GCP-Mamba"])

plt.figure(figsize=(8, 6))

models = ["BaseMamba", "GCP-Mamba (Permuted GO)", "GCP-Mamba"]
colors = ["#9E9E9E", "#FFB300", "#2196F3"]

for model, color in zip(models, colors):
    all_losses = []
    for seed in loss_history.keys():
        all_losses.append(loss_history[seed][model])
    all_losses = np.array(all_losses)
    
    mean_loss = all_losses.mean(axis=0)
    std_loss = all_losses.std(axis=0)
    
    x = np.arange(1, epochs + 1)
    plt.plot(x, mean_loss, label=model, color=color, linewidth=2)
    plt.fill_between(x, mean_loss - std_loss, mean_loss + std_loss, color=color, alpha=0.2)

plt.xlabel("Training Epochs")
plt.ylabel("Training Loss (MSE)")
plt.title("Model Convergence (Averaged over 5 Seeds)")
plt.legend()
plt.tight_layout()
plt.savefig("training_curve.png", dpi=300)
print("Saved training_curve.png")
