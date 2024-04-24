import matplotlib.pyplot as plt
from adjustText import adjust_text

models = [
    # "X3D-XS (baseline)", 
    # "X3D-XS", 
    # "Lighter X3D-XS", 
    # "MobileNetV3 (small)", 
    # "MobileNetV3 (small) + 0.3 Dropout",
    "MobileNetV3 (large)", 
    "X3D-M",
    "ResNet152",
    "S3D", 
    "Swin3D-S", 
    "Swin3D-S + 0.1 Dropout (all layers)",
    "Swin3D-S + 0.5 Dropout (last layer)", 
]
# 0.91, 0.91, 0.51, 0.06, 0.06, 
g_flops = [0.22, 6.72, 11.51, 43.88, 82.84, 82.84, 82.84]
# 3.79, 3.79, 3.79, 2.54, 2.54,
parameters = [5.48, 3.79, 60.19, 28.15, 49.82, 27.88, 49.82]
# 18.5, 28.6, 67.3, 36.8, 36.4, 
test_accuracy = [35.3, 65.2, 37.2, 38.8, 60.3, 55.8, 62.4]

plt.figure(figsize=(20, 14))
sizes = [200 * param for param in parameters]  # Scale bubble size by number of parameters
bubble = plt.scatter(g_flops, test_accuracy, s=sizes, c=parameters, cmap='viridis', alpha=0.6, edgecolors="w", linewidth=1.5)
plt.colorbar(bubble, label='Number of Parameters (Millions)')
plt.xlabel('GFLOPs', fontsize=18)
plt.ylabel('Test Accuracy (%)', fontsize=18)
plt.title('Bubble Plot of GFLOPs vs. Test Accuracy of Models', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Adding annotations and using adjust_text to avoid overlaps
texts = []
for i, txt in enumerate(models):
    texts.append(plt.text(g_flops[i], test_accuracy[i], txt, fontsize=18, ha='center'))
adjust_text(texts)

plt.grid(True)
plt.savefig("/ghome/group01/mcv-c6-2024-team1/Week6/plots/bubble_plot_revised.png", dpi=200)
