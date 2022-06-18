import matplotlib.pyplot as plt

tr_acc = [3.6656, 2.4938, 2.1059, 1.8024, 1.5476, 1.3205, 1.1204, 0.9404, 0.8149, 0.6990, 0.5969, 0.5246, 0.4752, 0.4135, 0.3875, 0.3585, 0.3277, 0.2984, 0.2707, 0.2690]

te_acc = [3.2295, 2.6236, 2.7024, 2.5090, 2.6123, 2.7911, 2.9816, 3.3626, 3.2205, 3.6323, 3.7371, 3.8889, 4.0336, 4.2986, 4.5681, 4.6837, 4.8695, 4.9375, 4.9694, 5.3434]

plt.plot(tr_acc, label='Train loss')
plt.plot(te_acc, label='Test loss')
plt.xlabel('Epochs')
plt.xticks(range(0, 21, 2))
plt.ylabel('Loss')
plt.legend()
plt.savefig("init_cnn_cifar100_loss.png")