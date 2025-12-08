import torch

class SmallMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        # self.fc = torch.nn.Linear(in_features=in_dim, out_features=out_dim)
        self.fc1 = torch.nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, input):
        input = input.view(input.size(0), -1) # First dim: batch size; Second dim: multiple of all others
        # print(input.size(0), input.size(1))
        hidden_rep = self.fc1(input)
        hidden_rep_relu = self.relu(hidden_rep)
        output = self.fc2(hidden_rep_relu)
        return output
        # return self.fc2(self.relu(self.fc1(input)))

if __name__ == '__main__':
    model = SmallMLP(in_dim=28*28, hidden_dim=128, out_dim=10)
    input = torch.randn(16, 28*28)
    logits = model(input)

    print("Input: ", input)
    print("Logits: ", logits)

    output_softmaxed = torch.softmax(logits, dim=1)
    print("Probabilities: ", output_softmaxed)

    print("Parameters of Network: ", model.parameters())