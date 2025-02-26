import matplotlib.pyplot as plt

# Data for rho and mAP (fixing alpha = 3.5)
rhos = [0.001, 0.003, 0.005, 0.007, 0.01, 0.02, 0.05]
mAP_values = [0.5823968503964475, 0.6236057558859616, 0.630684446313338,
              0.6322870674365071, 0.6182516321370617, 0.5664979271454987,
              0.46727585407535593]

# Find the index of the maximum mAP value
max_mAP_index = mAP_values.index(max(mAP_values))
optimal_rho = rhos[max_mAP_index]
optimal_mAP = mAP_values[max_mAP_index]

# Create the plot
plt.figure(figsize=(8, 5))

# Plot all points as stars ('*') in blue
plt.scatter(rhos, mAP_values, color='b', label="mAP vs Rho", marker='*')

# Highlight the maximum mAP value with a green star
plt.scatter(optimal_rho, optimal_mAP, color='g', s=100, zorder=5, marker='*')

# Add title and labels
plt.title("mAP vs Rho (Alpha = 3.5)")
plt.xlabel("Rho")
plt.ylabel("Mean Average Precision (mAP)")

# Adjust x-axis scale for better spacing
plt.xscale('log')
plt.xticks(rhos, labels=[str(r) for r in rhos])  # Showing specific rho values

# Add label with the optimal rho and mAP value
optimal_label = f"Optimal\nRho: {optimal_rho}\nmAP: {optimal_mAP:.4f}"
plt.text(optimal_rho * 1.1, optimal_mAP + 0.005, optimal_label, color='g', ha='left', va='top', fontsize=8)

# Adjust layout for better spacing
plt.tight_layout()

# Save the plot as an image
plt.savefig("mAP_vs_Rho_plot_with_optimal_label.png", dpi=300)

# Show the plot
plt.show()
