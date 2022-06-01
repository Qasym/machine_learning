### Intorduction
- In this project I analyzed the performance of multilayer perceptron.<br>
I used very simple structure that was inspired from 3blue1brown's video in youtube<br>
The structure of my neural network(NN) looks like this:<br>
*input_layer* - *hidden_layer* - *output_layer*<br>
*2 : 250 : 1* (layers are shown respectively to the above structure)
- The hyperparameters are:
    - Learning rate: **0.1**
    - Learning rate scheduling scheme: **constant learning rate** 
    - Number of hidden layer nodes: **250**
    - Epochs: **500**

### First analysis
- In the first part of the analysis I developed a NN that was trained on<br>
the whole training set provided in [Trn.csv](./Trn.csv) and tested it<br>
using the test set provided in [Tst.csv](./Tst.csv)
- Results:
    - Train loss: `0.05`
    - Train accuracy: `97.94%`
    - Test loss: `0.09`
    - Test accuracy: `97.46%`
- You can also refer to the [NN_all_accuracy.png](./NN_all_accuracy.png) to see<br>
how accuracy was chaning for the train and the test set as we pass epochs
- Similarly you can refer to the [NN_all_loss.png](./NN_all_loss.png) to see<br>
how loss was dropping down after only about 10 epochs!
- By using the NN in this part, I obtained the graph for the [Grid.csv](./Grid.csv)<br>
Take a look at [NN_all_grid.png](./NN_all_grid.png)<br> 
We can observe how it draws the spiral figure

### Second analysis
- In this part of the analysis I developed a NN that was trained on<br>
the first 40 entries of the training set [Trn.csv](./Trn.csv) and tested it<br>
using the test set provided in [Tst.csv](./Tst.csv)
- Results:
    - Train loss: `0.00`
    - Train accuracy: `100.00%`
    - Test loss: `1.09`
    - Test accuracy: `84.60%`
- As you can see the train accuracy and test accuracy differ significantly<br>
as compared to the first analysis. That is the case of data overfitting, it<br>
happens due to not giving our model enough data to train on, it learns how to solve<br>
"home" problems, but fails to provide good performance in "real" problems
- We can see how the [NN_40_accuracy.png](./NN_40_accuracy.png) differs from the<br>
first analysis
- We can also see how [NN_40_loss.png](./NN_40_loss.png) differs from the first analysis
- Drawing a grid from [Grid.csv](./Grid.csv) gives as inaccurate picture<br> 
as shown in the [NN_40_grid.png](./NN_40_grid.png)