from collections import defaultdict
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from .MiscUtils import sample, flatten

def get_fewshot_dataset(dataset, n_way=5, n_shot=5, classes=None, seed=0):
    """Returns a Subset of [dataset] giving a n-shot n-way task.

    Args:
    dataset -- ImageFolder-like dataset
    n_way   --
    n_shot
    classes --
    """
    use_all_list = ["all", -1]
    
    if classes in use_all_list and n_shot in use_all_list:
        return dataset

    if classes in use_all_list:
        classes = set(dataset.classes)
    elif classes is None:
        classes = set(sample(dataset.classes, k=n_way, seed=seed))
    else:
        classes = set(classes)

    classes = {dataset.class_to_idx[c] for c in classes}
    class2idxs = defaultdict(lambda: [])
    for idx,t in enumerate(dataset.targets):
        if t in classes:
            class2idxs[t].append(idx)

    if not n_shot in use_all_list:
        try:
            class2idxs = {c: sample(idxs, k=n_shot, seed=seed)
                for c,idxs in class2idxs.items()}
        except ValueError as e:
            class2n_idxs = "\n".join([f"\t{c}: {len(idxs)}"
                for c,idxs in class2idxs.items()])
            tqdm.write(f"Likely --val_n_shot asked for more examples than are available | val_n_shot {n_shot} | class to num idxs: {class2n_idxs}")
            raise e
  
    indices = flatten([idxs for idxs in class2idxs.values()])
    dataset = Subset(dataset, indices=indices)
    class2idx = {c: idx for idx,c in enumerate(sorted(classes))}
    return XYDataset(dataset,
        target_transform=lambda c: class2idx[c],
        classes=classes)    
        
class XYDataset(Dataset):
    """A simple dataset returning examples of the form (transform(x), y)."""

    def __init__(self, data, transform=None, target_transform=None, 
        normalize=False, classes=None):
        """Args:
        data        -- a sequence of (x,y) pairs
        transform   -- the transform to apply to each returned x-value
        """
        super(XYDataset, self).__init__()
        self.data = data
        self.transform = transform
        self.target_transform = target_transform
        self.normalize = normalize
        
        if self.normalize:
            means, stds = get_image_channel_means_stds(XYDataset(self.data))
            self.transform = transforms.Compose([transform,
                transforms.Normalize(means, stds, inplace=True)])

        if hasattr(self.data, "class_to_idx"):
            self.classes = list(self.data.class_to_idx.keys())
            self.class_to_idx = deepcopy(self.data.class_to_idx)
        else:
            pass

        if hasattr(self.data, "classes"):
            self.classes = deepcopy(self.data.classes)
        elif not classes is None:
            self.classes = classes
        else:
            loader = DataLoader(self.data, batch_size=128, num_workers=8)
            self.classes = set()
            for _,y in tqdm(loader,
                desc="Getting XYDataset class information",
                leave=False):
                self.classes |= set(y.tolist())
            
            self.classes = list(sorted(self.classes))

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = x if self.transform is None else self.transform(x)
        y = y if self.target_transform is None else self.target_transform(y)
        return x,y

def get_image_channel_means_stds(dataset, args):
    """Returns an (mu, sigma) tuple where [mu] and [sigma] are tensors in which
    the ith element gives the respective ith mean and standard deviation of the
    ith channel of images in [dataset].

    Args:
    dataset -- dataset returning (x,y) pairs with [x] a CxHxW tensor
    bs      -- batch size to use in the computation
    """
    loader = DataLoader(dataset,
        num_workers=args.num_workers,
        batch_size=1024,
        pin_memory=True)

    means, stds = torch.zeros(3, device=device), torch.zeros(3, device=device)
    for x,_ in tqdm(loader,
        desc="PREPARING DATA: Finding image channel stats for standardication",
        dynamic_ncols=True,
        leave=False):

        bs, c, _, _ = x.shape
        x = x.to(device, non_blocking=True)
        means += torch.mean(x, dim=[0, 2, 3]) * bs
        stds += torch.std(x, dim=[0, 2, 3]) * bs

    means = means.cpu() / len(dataset)
    stds = stds.cpu() / len(dataset)
    tqdm.write(f"LOG: Found images means {means.tolist()} and stds {stds.tolist()}")
    return means, stds