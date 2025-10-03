import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ------------------ Parseval Update ------------------
def parseval_update(model, beta=0.0003, num_passes=1):
    """
    Apply Parseval tight frame retraction update:
        W <- (1 + beta)W - beta W(W^T W)
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            W = module.weight.data
            shape = W.shape
            W_mat = W.view(shape[0], -1)   # flatten (out_channels, in_dim)

            for _ in range(num_passes):
                WT_W = torch.mm(W_mat.t(), W_mat)  # (in_dim, in_dim)
                W_mat = (1 + beta) * W_mat - beta * torch.mm(W_mat, WT_W)

            module.weight.data = W_mat.view(shape)

# ------------------ WideResNet definition ------------------
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.dropRate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.shortcut = (None if self.equalInOut else
                         nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                   padding=0, bias=False))

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu(self.bn1(x))
        else:
            out = self.relu(self.bn1(x))
        out = self.conv1(out if self.equalInOut else x)
        out = self.relu(self.bn2(out))
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)
        out = self.conv2(out)
        return out + (x if self.equalInOut else self.shortcut(x))

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes,
                                out_planes,
                                i == 0 and stride or 1,
                                dropRate))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes=10, dropRate=0.3):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth-4)%6 == 0)
        n = (depth-4)//6
        block = BasicBlock

        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

# ------------------ Training Script ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.0003,
                        help="Parseval retraction step size")
    parser.add_argument("--num_passes", type=int, default=1,
                        help="Number of retraction passes per step")
    parser.add_argument("--restart", action="store_true",
                        help="Ignore checkpoints and start fresh")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    if device == 'cuda':
        print("GPU:", torch.cuda.get_device_name(0))

    # CIFAR-10 data loaders
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
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                             shuffle=False, num_workers=8, pin_memory=True)

    # Model, loss, optimizer, scheduler
    model = WideResNet(depth=28, widen_factor=10, num_classes=10, dropRate=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    best_test_acc = 0.0

    # Resume
    if os.path.exists("wrn28_10_last.pt") and not args.restart:
        print("Resuming from checkpoint: wrn28_10_last.pt")
        checkpoint = torch.load("wrn28_10_last.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_test_acc = checkpoint["best_acc"]
    else:
        if args.restart:
            print("Restart flag set → ignoring checkpoints")
        start_epoch = 0
        best_test_acc = 0.0

    @torch.no_grad()
    def eval_acc():
        model.eval()
        correct, total = 0, 0
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast():
                preds = model(x).argmax(1)
            total += y.size(0)
            correct += (preds == y).sum().item()
        return 100.0 * correct / total

    # Train for N epochs
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # ✅ Apply Parseval update
            parseval_update(model, beta=args.beta, num_passes=args.num_passes)

            running_loss += loss.item()
            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        train_acc = 100.*correct/total
        test_acc = eval_acc()
        print(f"Epoch {epoch+1}/{args.epochs}: Train {train_acc:.2f}% | Test {test_acc:.2f}% | Loss {running_loss/len(trainloader):.3f}")
        scheduler.step()

        # Save checkpoints
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_acc": best_test_acc,
        }, "wrn28_10_last.pt")

        if (epoch+1) % 50 == 0:
            ckpt_name = f"wrn28_10_epoch{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_name)
            print(f"Saved checkpoint: {ckpt_name}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "wrn28_10_best.pt")
            print(f"New best model saved with Test Acc = {best_test_acc:.2f}%")

    torch.save(model.state_dict(), "wrn28_10_final.pt")
    print("Training finished. Saved final model: wrn28_10_final.pt")

if __name__ == "__main__":
    main()
