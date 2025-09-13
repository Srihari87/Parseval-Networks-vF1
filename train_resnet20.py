import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models.resnet20 import resnet20   # ✅ make sure you have the architecture file in models/

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    if device == 'cuda':
        print("GPU:", torch.cuda.get_device_name(0))

    # ------------------ Data ------------------
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.243, 0.261)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.243, 0.261)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=4, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                             shuffle=False, num_workers=4, pin_memory=True)

    # ------------------ Model ------------------
    model = resnet20().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[60, 120, 160], gamma=0.2)

    start_epoch = 0
    best_test_acc = 0.0

    # ------------------ Resume ------------------
    if os.path.exists("resnet20_last.pt"):
        print("Resuming from checkpoint: resnet20_last.pt")
        checkpoint = torch.load("resnet20_last.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_test_acc = checkpoint["best_acc"]

    # ------------------ Training ------------------
    for epoch in range(start_epoch, 200):
        model.train()
        correct, total, running_loss = 0, 0, 0

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total

        # ---- Validation ----
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        test_acc = 100. * correct / total

        print(f"Epoch {epoch+1}/200: "
              f"Train {train_acc:.2f}% | "
              f"Test {test_acc:.2f}% | "
              f"Loss {running_loss/len(trainloader):.3f}")

        scheduler.step()

        # ---- Save last checkpoint ----
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_acc": best_test_acc,
        }, "resnet20_last.pt")

        # ---- Save every 50 epochs ----
        if (epoch+1) % 50 == 0:
            ckpt_name = f"resnet20_epoch{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_name)
            print(f"Saved checkpoint: {ckpt_name}")

        # ---- Save best model ----
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "resnet20_best.pt")
            print(f"New best model saved with Test Acc = {best_test_acc:.2f}%")

    # ---- Final save ----
    torch.save(model.state_dict(), "resnet20_final.pt")
    print("Training finished. Saved final model: resnet20_final.pt")

if __name__ == "__main__":
    main()
