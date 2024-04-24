import matplotlib.pyplot as plt
import numpy as np 

# Model names
models = ['X3D-XS (baseline)', 'X3D-XS', 'MobileNetV3 (small) + DP 0.3', 'MobileNetV3 (small)', 'Lighter X3D-XS']
# models = ['X3D-XS (baseline)','X3D-M', 'MobileNetV3 (large)', 'Swin3D-S', 'Swin3D-S + 0.5 DP', 'Swin3D-S + 0.1 DP', 'S3D', 'ResNet152']

# Train accuracy values
train_accuracy = [26.9, 36.4, 79.51, 84.739, 92.02]
#train_accuracy = [26.9, 92.74, 91.25, 96.3, 94.8, 93.4, 88.6, 94.97]       

# Validation accuracy values
val_accuracy = [15.7, 27.3, 34.5, 31.14, 57.67]
#val_accuracy = [15.7, 66.94, 27.96, 53.2, 58.1, 52.1, 32.2, 33.47]

test_accuracy= [18.5, 28.6, 36.4, 36.8, 60.3]
#test_accuracy = [18.5,65.2, 35.3, 60.3, 62.4, 55.8, 38.8, 37.2]
# Number of models
num_models = len(models)

# Set the width of the bars
bar_width = 0.2

# Set the x locations for the groups
index = np.arange(num_models)

# Plotting
plt.figure(figsize=(12, 8))

train_bars = plt.bar(index, train_accuracy, bar_width, label='Train Accuracy')
val_bars = plt.bar(index + bar_width, val_accuracy, bar_width, label='Validation Accuracy')
test_bars = plt.bar(index + 2 * bar_width, test_accuracy, bar_width, label='Test Accuracy')

plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.title('Model Train, Validation, and Test Accuracy')
plt.xticks(index + bar_width, models, rotation=45, ha='right')
plt.legend()

# Show plot
plt.tight_layout()
plt.show()
plt.savefig('plots/barplots_11.jpg')
