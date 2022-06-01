# Report


## Hyperparameters

1. Learning rate is: `0.1`
2. Number of nodes in the hidden layer is: `250`
3. Number of hidden layers: `1`
    - The structure looks like this:<br>
        **input_layer**(2 nodes) - **hidden_layer**(250 nodes) - **output_layer**(1 node)
4. Number of epochs is: `500`


## Training accuracy and loss

- For the Neural Network(NN) that was trained on the whole training set<br>
I obtained the following results:<br>
![Percentage_all](./all_accuracy_loss.png)<br>
![Graph_accuracy_all](./train_accuracy_all.png)<br>
![Graph_loss_all](./train_loss_all.png)<br>
As for the grid, it looks like this:<br>
![Grid_all](./gridAll.png)<br>

- For the NN that was trained only on the first 40 entries of the training set<br>
I obtained the following results:<br>
![Percentage_40](./40_accuracy_loss.png)<br>
![Graph_accuracy_40](./train_accuracy_40.png)<br>
![Graph_loss_40](./train_loss_40.png)<br>
As for the grid it looks like this:<br>
![Grid_40](./grid40.png)<br>