from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


tb_writer = SummaryWriter("logs/images_test")
if __name__ == "__main__":
    dataset = datasets.CIFAR10('./data', download=False, train=True, transform=transforms.ToTensor())
    dataLoader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=True, num_workers=2)
    idx = 0
    for batch in dataLoader:
        images, targets = batch
        tb_writer.add_images("test_droplast", images, idx)
        idx += 1

    tb_writer.close()