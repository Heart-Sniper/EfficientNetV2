import torch
from tqdm import tqdm
import sys

from torch.utils.data import Dataset
from PIL import Image


class MyDataSet(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img_path = self.images_path[idx]
        label = self.images_class[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label, img_path

    @staticmethod
    def collate_fn(batch):
        images, labels, img_paths = zip(*batch)

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)

        return images, labels, img_paths


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)

    sample_num = 0
    misclassified_images = []

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels, img_paths = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        # Track misclassified image paths
        misclassified_indices = torch.nonzero(pred_classes != labels.to(device)).squeeze()

        # Check if there are misclassified samples
        if misclassified_indices.ndimension() == 0:
            misclassified_indices = misclassified_indices.unsqueeze(0)

        for mis_idx in misclassified_indices:
            misclassified_images.append((img_paths[mis_idx.item()], pred_classes[mis_idx.item()].item()))

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, misclassified_images
