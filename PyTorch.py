import math
import data_loader as dl
import torch
import torch.nn as nn

input_size = 784
hidden_size = 16
num_classes = 10
num_epochs = 10
batch_size = 50
learning_rate = 0.001


class NeuralNetwork(nn.Module):
    def __init__(self, input_layer, hidden_layer, output_layer):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_layer, hidden_layer)
        self.activation1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_layer, hidden_layer)
        self.activation2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_layer, output_layer)
        self.activation3 = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.l1(x)
        out = self.activation1(out)
        out = self.l2(out)
        out = self.activation2(out)
        out = self.l3(out)
        out = self.activation3(out)
        return out


def Prediction(output):
    result = torch.zeros((len(output), num_classes))
    for index in range(len(output)):
        digit = torch.argmax(output[index])
        result[index][digit] = 1
    return result


if __name__ == "__main__":
    device = torch.device('cpu')
    data = dl.smooth_data()

    train_images = data[0]
    train_labels = data[1]
    train_images = torch.tensor(train_images, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)

    records = len(train_labels)
    num_batches = math.floor(records / batch_size)

    model = NeuralNetwork(input_size, hidden_size, num_classes)

    criteria = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i in range(num_batches):
            input_batch = train_images[i * batch_size:(i + 1) * batch_size]
            label_batch = train_labels[i * batch_size:(i + 1) * batch_size]
            input_batch = input_batch.to(device)
            label_batch = label_batch.to(device)

            outputs = model(input_batch)
            loss = criteria(outputs, label_batch)
            print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    test_images = data[2]
    test_labels = data[3]
    test_images = torch.tensor(test_images, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)
    test_images = test_images.to(device)
    test_labels = test_labels.to(device)

    predictions = Prediction(model(test_images))
    true_predictions = 0
    for i in range(len(test_labels)):
        if torch.equal(test_labels[i], predictions[i]):
            true_predictions += 1
    accuracy = (true_predictions / len(test_labels)) * 100
    print(accuracy)
