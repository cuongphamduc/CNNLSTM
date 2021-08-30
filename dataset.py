import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F


class FireDataset(Dataset):
    def __init__(self, root, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])):
        self.root = root
        self.transform = transform
        self.all_class = os.listdir(root)

        self.x = []
        self.y = []

        for (aa, i) in enumerate(self.all_class):
            path = os.path.join(self.root, i)

            all_images = os.listdir(path)

            for j in range(0, len(all_images), 15):
                images = torch.tensor(())

                for k in all_images[j: j + 15]:
                    img = Image.open(os.path.join(path, k)).convert('RGB').resize((224, 224))
                    img = self.transform(img)
                    img = Variable(torch.unsqueeze(img, 0))

                    images = torch.cat((images, img), 0)

                self.x.append(images)
                self.y.append(aa)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        images = self.x[index]
        label = torch.tensor(self.y[index])

        return images, label
