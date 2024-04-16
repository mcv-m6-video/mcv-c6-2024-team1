""" This module contains classes and functions for statistics. """
class RollingMean:
    def __init__(self, window_size):
        self.window_size = window_size
        self.data = []

    def __call__(self, value):
        self.data.append(value)
        if len(self.data) > self.window_size:
            self.data.pop(0)
        
        if len(self.data) == 0:
            return None
        return sum(self.data) / len(self.data)

def evaluate_per_class_accuracy(
        model: nn.Module, 
        valid_loader: DataLoader, 
        device: str,
        description: str = ""
    ) -> np.array:
    """
    Evaluates the per-class accuracy of the given model using the provided data loader.

    Args:
        model (nn.Module): The neural network model to be validated.
        valid_loader (DataLoader): The data loader containing the validation dataset.
        device (str): The device on which the model and data should be processed ('cuda' or 'cpu').
        description (str, optional): Additional information for tracking epoch description during training. Defaults to "".

    Returns:
        np.array: Array of per-class accuracies.
    """
    model.eval()
    pbar = tqdm(valid_loader, desc=description, total=len(valid_loader))
    class_correct = np.zeros(valid_loader.dataset.get_num_classes())
    class_total = np.zeros(valid_loader.dataset.get_num_classes())
    for batch in pbar:
        clips, labels = batch['clips'].to(device), batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(clips)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    per_class_accuracy = class_correct / class_total
    return per_class_accuracy