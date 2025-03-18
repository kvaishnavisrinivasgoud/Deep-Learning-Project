# Deep-Learning-Project

## Install Required Libraries:

-Make sure you have TensorFlow installed. You can install it using pip:

              pip install tensorflow matplotlib



## Output:

1.The model will train for 10 epochs and display the training/validation accuracy and loss curves.

2.A confusion matrix will show how well the model performs for each class.

3.The model will be saved as cifar10_cnn_model.h5.



## Example Results

1.Test Accuracy: ~68-70% (may vary slightly due to random initialization).

2.Training Accuracy: ~75% (higher than test accuracy, indicating some overfitting).

3.Visualizations: Graphs for accuracy/loss and a confusion matrix.



## Improvements:

-To improve the results:

1.Data Augmentation: Apply transformations like rotation, flipping, and cropping to the training data.

2.Deeper Model: Use a more complex architecture like ResNet or VGG.

3.Regularization: Add dropout layers or L2 regularization to reduce overfitting.

4.More Epochs: Train for more epochs (e.g., 20-30).
