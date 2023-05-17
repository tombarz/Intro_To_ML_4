import backprop_data

import backprop_network
import matplotlib.pyplot as plt


training_data, test_data = backprop_data.load(train_size=10000,test_size=5000)
learning_rate = 0.001
figure, axis = plt.subplots(3)


# Combine all the operations and display

net = backprop_network.Network([784, 40, 10])
for i in range(7):
    x, training_accuracy, test_accuracy, loss = net.SGD_for_plotting(training_data, epochs=30, mini_batch_size=10, learning_rate=learning_rate, test_data=test_data)
    axis[0].plot(x, training_accuracy, label='learning rate = {0}'.format(learning_rate))
    axis[1].plot(x, test_accuracy, label='learning rate = {0}'.format(learning_rate))
    axis[2].plot(x, loss, label='learning rate = {0}'.format(learning_rate))
    learning_rate = learning_rate * 10
axis[0].set_title('Training Accuracy')
axis[1].set_title('Test Accuracy')
axis[2].set_title('Loss')
axis[0].set_xlabel('Epochs')
axis[1].set_xlabel('Epochs')
axis[2].set_xlabel('Epochs')
axis[0].set_ylabel('Accuracy')
axis[1].set_ylabel('Accuracy')
axis[2].set_ylabel('Loss')
axis[0].legend()
axis[1].legend()
axis[2].legend()
axis[2].set_ylim([0, 70])
plt.show()


