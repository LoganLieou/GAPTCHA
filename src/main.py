import torch
from model import Network

def main():
    model = Network()
    print(model)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

if __name__ == "__main__":
    main()
