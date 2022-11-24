from collections import defaultdict
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import h5py

from .MiscUtils import sample, flatten
def get_fewshot_dataset(dataset, n_way=5, n_shot=5, classes=None, seed=0,   
    fewer_shots_if_needed=False):
    """Returns a Subset of [dataset] giving a n-shot n-way task.

    Args:
    dataset                 -- ImageFolder-like dataset
    n_way                   -- number of classes to use
    n_shot                  -- number of shots to use
    classes                 -- classes to use (overrides [n_way])
    fewer_shots_if_needed   -- if [dataset] doesn't have all the [n_shots] for a
                                class, use less than [n_shots]
    """
    use_all_list = ["all", -1]
    
    if classes in use_all_list and n_shot in use_all_list:
        return dataset

    if classes in use_all_list:
        classes = set(dataset.classes)
    elif classes is None:
        n_way = len(dataset.classes) if n_way in use_all_list else n_way
        classes = set(sample(dataset.classes, k=n_way, seed=seed))
    else:
        classes = set(classes)

    classes = {dataset.class_to_idx[c] for c in classes}
    class2idxs = defaultdict(lambda: [])
    for idx,t in enumerate(dataset.targets):
        if t in classes:
            class2idxs[t].append(idx)

    if not n_shot in use_all_list:
        n_shot_fn = lambda x: (min(len(x), n_shot) if fewer_shots_if_needed else n_shot)
        try:
            class2idxs = {c: sample(idxs, k=n_shot_fn(idxs), seed=seed)
                for c,idxs in class2idxs.items()}
        except ValueError as e:
            class2n_idxs = "\n".join([f"\t{c}: {len(idxs)}"
                for c,idxs in class2idxs.items()])
            tqdm.write(f"Likely --val_n_shot asked for more examples than are available | val_n_shot {n_shot} | class to num idxs: {class2n_idxs}")
            raise e
  
    indices = Misc.flatten([idxs for idxs in class2idxs.values()])
    return ImageFolderSubset(dataset, indices=indices)

class ImageFolderSubset(Dataset):
    """Subset of an ImageFolder that preserves key attributes. Besides preserving ImageFolder-like attributes, the key improvement over a regular Subset is a target2idx dictionary that maps a target returned from [data] to a number in
    [0, len(classes)) which is necessary for classification.

    Doing this efficiently is oddly non-trivial.

    Besides maintaining [targets], [classes] and [class_to_idx] attributes,
    there are several key constraints:
    1) Every element of the [class_to_idx.values()] is a member of [targets]. As
        [targets] are integers in [0...N-1], this means that neither [targets]
        nor [class_to_idx] this attribute is not preserved by constructing an ImageFolderSubset
    2) Constructing this subset yields a dataset whose [classes] attribute is a
        subset of the same attribute of [data]

    With this constraint met, it's possible to construct this dataset on top of
    itself any number of times.

    Args:
    data    -- ImageFolder-like dataset
    indices -- list giving subset indices
    """

    def __init__(self, data, indices):
        super(ImageFolderSubset, self).__init__()
        self.data = data
        self.root = self.data.root
        self.indices = indices

        idxs_set = set(indices)

        # Mapping from indices we care about to the targets they have in [data]
        data_idx2target = {idx: t
            for idx,t in enumerate(data.targets)
            if idx in idxs_set}
        
        # Unique targets in subset of data
        data_targets = set(data_idx2target.values())

        # Mapping from unique targets in subset of data to their class
        data_target2class = {t: c for c,t in data.class_to_idx.items()
            if t in data_targets}

        # Mapping from indices we care about to their classes
        data_idx2class = {idx: data_target2class[t]
            for idx,t in enumerate(data.targets)
            if idx in idxs_set}

        # Subset of the classes in [data]
        self.classes = set(data_target2class.values())
        self.class_to_idx = {c: idx for idx,c in enumerate(sorted(self.classes))}
        self.data_target2idx = {t: idx for idx,t in enumerate(sorted(data_targets))}
        self.targets = [self.data_target2idx[t] for t in data_idx2target.values()]

    def __str__(self): return f"{self.__class__.__name__} [root={self.root} | length={self.__len__()}]"

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        x,y = self.data[self.indices[idx]]
        return x, self.data_target2idx[y]

def get_image_channel_means_stds(dataset, num_workers=8):
    """Returns an (mu, sigma) tuple where [mu] and [sigma] are tensors in which
    the ith element gives the respective ith mean and standard deviation of the
    ith channel of images in [dataset].

    Args:
    dataset -- dataset returning (x,y) pairs with [x] a CxHxW tensor
    bs      -- batch size to use in the computation
    """
    loader = DataLoader(dataset,
        num_workers=num_workers,
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
