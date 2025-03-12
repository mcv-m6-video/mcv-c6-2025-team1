import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Data
categories = ['Mean', 'Max', 'Median']
x_labels = ['0', '0.2', '0.4', '0.5', '0.6', '0.8', '1']

hota_values = [
    [15.91, 61.75, 72.93, 82.02, 85.42, 86.67, 88.66],  # Mean
    [19.46, 61.93, 72.29, 79.48, 80.10, 87.28, 88.66],  # Max
    [16.47, 63.69, 72.12, 81.64, 85.44, 86.66, 88.66]   # Median
]

idf1_values = [
    [12.15, 66.87, 73.77, 88.85, 92.24, 92.63, 93.76],  # Mean
    [16.55, 65.68, 74.82, 85.94, 85.80, 93.76, 93.76],  # Max
    [12.15, 71.85, 74.58, 87.42, 92.21, 92.62, 93.76]    # Median
]

''' For SORT+OF
SORT:
Mean:
HOTA = [15.91, 61.75, 72.93, 82.02, 85.42, 86.67, 88.66]
IDF1 = [12.15, 66.87, 73.77, 88.85, 92.24, 92.63, 93.76]

Max:
HOTA = [19.46, 61.93, 72.29, 79.48, 80.10, 87.28, 88.66]
IDF1 = [16.55, 65.68, 74.82, 85.94, 85.80, 93.76, 93.76]

Median:
HOTA = [16.47, 63.69, 72.12, 81.64, 85.44, 86.66, 88.66]
IDF1 = [12.15, 71.85, 74.58, 87.42, 92.21, 92.62, 93.76]

STRONG-SORT

hota_values = [
    [14.69, 72.71, 83.64, 83.25, 86.37, 86.73, 88.99],  # Mean
    [15.67, 73.26, 84.08, 86.08, 85.47, 86.18, 88.99],  # Max
    [14.44, 70.23, 83.64, 85.65, 86.35, 86.74, 88.99]   # Median
]
idf1_values = [
    [9.29, 83.39, 90.93, 89.12, 92.67, 92.39, 93.25],  # Mean
    [9.56, 79.71, 90.74, 92.34, 90.74, 90.92, 93.25],  # Max
    [8.32, 77.5, 90.56, 92.61, 92.67, 92.43, 93.25]    # Median
]
'''

# Plot HOTA heatmap
plt.figure(figsize=(10, 4))
sns.heatmap(hota_values, annot=True, fmt=".2f", cmap="Reds", xticklabels=x_labels, yticklabels=categories)
plt.xlabel("Alpha")
plt.ylabel("Fusion Metrics")
plt.title("HOTA Heatmap")
plt.savefig("hota_heatmap_sort.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot IDF1 heatmap
plt.figure(figsize=(10, 4))
sns.heatmap(idf1_values, annot=True, fmt=".2f", cmap="Reds", xticklabels=x_labels, yticklabels=categories)
plt.xlabel("Alpha")
plt.ylabel("Fusion Metrics")
plt.title("IDF1 Heatmap")
plt.savefig("idf1_heatmap-sort.png", dpi=300, bbox_inches='tight')
plt.close()


