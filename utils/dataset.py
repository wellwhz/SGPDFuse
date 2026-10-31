import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class FusionData(Dataset):
    def __init__(self, root, a='ir', b='vi', ca=1, cb=3):
        self.da = os.path.join(root, a)
        self.db = os.path.join(root, b)
        self.ma = 'L' if ca == 1 else 'RGB'
        self.mb = 'L' if cb == 1 else 'RGB'
        self.names = sorted(set(os.listdir(self.da)) & set(os.listdir(self.db)))
        w, h = Image.open(os.path.join(self.da, self.names[0])).size
        self.size = (w // 16 * 16, h // 16 * 16)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        a = Image.open(os.path.join(self.da, n)).convert(self.ma).resize(self.size, Image.BILINEAR)
        b = Image.open(os.path.join(self.db, n)).convert(self.mb).resize(self.size, Image.BILINEAR)
        return to_tensor(a), to_tensor(b)
