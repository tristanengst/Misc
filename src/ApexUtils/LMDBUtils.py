"""Source code for working with LMDB files.

USAGE:
python UtilsLMDB
    --data_dirs containers_of_imagefolders_to_convert # accepts glob!
    --enclosing_dir directory_to_write_outputs_inside_with_file_structure_of_input
    --new_name replace_the_common_substring_in_the_basenames_of_folders_in_--data_dirs_with_this

"""
import argparse
import numpy as np
import os
from tqdm import tqdm
import lmdb
from torch.utils.data import Dataset
from PIL import Image
import shutil
import glob

def lmdb_file_contains_augs(lmdb_file):
    """Returns if LMDB file [lmdb_file] contains images that should be
    interpreted as augmentations rather than images in their own right.
    """
    env = lmdb.open(lmdb_file, readonly=True, lock=False, readahead=False,
        meminit=False)
    with env.begin(write=False) as txn:
        keys = txn.get("keys".encode("ascii"))
        keys = [k for k in keys.decode("ascii").split(",")]

    return any(["_aug" in k for k in keys])

def longest_common_starting_substring(l):
    """Returns the longest common substring in strings in list [l], with all
    substrings starting at the beginning.
    """
    min_length = min([len(x) for x in l])
    for idx in range(min_length):
        substring = l[0][:idx]
        if all([x.startswith(substring) for x in l]):
            continue
        else:
            return l[0][:max(0, idx - 1)]

    return l[0]

def is_image(f):
    """Returns if file [f] is an image."""
    return any([f.lower().endswith(e) for e in [".png", ".jpg", ".jpeg"]])

def _read_lmdb_img(env, key):
    """Returns a PIL image from [key] in opened LMDB file [env]."""
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode("ascii"))
        buf_dims = txn.get(f"{key}.dims".encode("ascii")).decode("ascii")
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    h, w, c = [int(s) for s in buf_dims.split()]
    img = img_flat.reshape(h, w, c)
    return Image.fromarray(img)

def _write_lmdb_img(env, image):
    """Writes the image at [image] to LMDB dataset open for writing with
    [env], and returns the key under which the image can be found in the LMDB
    dataset.
    """
    with env.begin(write=True) as txn:
        img_name = os.path.basename(image)
        img_dir = os.path.dirname(image)
        key = f"{os.path.basename(img_dir)}/{img_name}"

        image = np.array(Image.open(image), dtype=np.uint8)
        dims = " ".join([f"{d:d}" for d in image.shape])
        txn.put(key.encode("ascii"), image)
        txn.put(f"{key}.dims".encode("ascii"), dims.encode("ascii"))

    return key

class LMDBImageFolder(Dataset):
    """A near drop-in replacement for an ImageFolder, but reads from a single
    LMDB file. The idea is that instead of creating an ImageFolder over
    something, one can check if the thing ends with '.lmdb' and construct this
    if that is the case.

    Args:
    source              -- LMDB file equivalent to one ImageFolder dataset
    transform           -- transform for images
    target_transform    -- transform for targets
    """
    def __init__(self, source, transform=None, target_transform=None):
        super(LMDBImageFolder, self).__init__()

        self.env = lmdb.open(source, readonly=True, lock=False, readahead=False,
            meminit=False)

        with self.env.begin(write=False) as txn:
            keys = txn.get("keys".encode("ascii"))
            keys = sorted([k for k in keys.decode("ascii").split(",")])

        self.classes = sorted({os.path.dirname(k) for k in keys})
        self.class2idx = {cls: idx for idx,cls in enumerate(self.classes)}
        self.samples = [(k, self.class2idx[os.path.dirname(k)]) for k in keys]
        self.targets = [s[1] for s in self.samples]

        self.transform = transform
        self.target_transform = target_transform

        tqdm.write(f"LOG: Constructed LMDBImageFolder over {source}")

    def loader(self):
        def _loader(key):
            with env.begin(write=False) as txn:
                buf = txn.get(key.encode("ascii"))
                buf_dims = txn.get(f"{key}.dims".encode("ascii")).decode("ascii")
            img_flat = np.frombuffer(buf, dtype=np.uint8)
            h, w, c = [int(s) for s in buf_dims.split()]
            img = img_flat.reshape(h, w, c)
            return Image.fromarray(img)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        key, y = self.samples[idx]

        # Function inlining to increase speed. Equivalent to _read_lmdb_img().
        with self.env.begin(write=False) as txn:
            buf = txn.get(key.encode("ascii"))
            buf_dims = txn.get(f"{key}.dims".encode("ascii")).decode("ascii")
        img_flat = np.frombuffer(buf, dtype=np.uint8)
        h, w, c = [int(s) for s in buf_dims.split()]
        x = img_flat.reshape(h, w, c)
        x = Image.fromarray(x)

        x = x if self.transform is None else self.transform(x)
        y = y if self.target_transform is None else self.transform(y)
        return x, y

def lmdb_to_image_folder(source, out_path):
    """Creates an ImageFolder compatible directory with name [out_path] from the
    files in LMDB file [source].
    """
    out_path = out_path.rstrip("/")
    dataset = LMDBImageFolder(source)

    for idx,(k,_) in tqdm(enumerate(dataset.samples),
        desc="Writing LMDB dataset to ImageFolder",
        dynamic_ncols=True,
        total=len(dataset)):

        image_class = os.path.basename(os.path.dirname(k)).strip("/")
        image_name = os.path.basename(k)
        if not os.path.exists(f"{out_path}/{image_class}"):
            os.makedirs(f"{out_path}/{image_class}")
        dataset[idx][0].save(f"{out_path}/{image_class}/{image_name}")

def image_folder_to_lmdb(source, out_path, fix_images=True, res=None):
    """Creates an LMDB file encoding directory [source] such that an ImageFolder
    created over [source] would be equivalent to an LMDBImageFolder created over
    the LMDB file.

    Args:
    source      -- directory that could be read as an ImageFolder
    out_path    -- path to write LMDB dataset to
    fix_images  -- whether to fix images that have one or more predefined errors
    res         -- integer or (integer, interger) tuple to resize images to. If
                    not specified, all images must have the same resolution.
    """
    def fix_image(image, fixed_image_dir, res=None):
        """If there is a problem with the image at [image], saves a fixed
        version of it to a file under [fixed_images_dir]. In the former case,
        returns [image]; in the latter returns the path to which the fixed
        image was saved to.

        In no case will the original image be modified in any way, ie. the fixes
        will be reflected in only the LMDB file.

        Args:
        image          -- path to image to be potentially fixed
        fixed_image_dir    -- path to folder where fixed images will be saved
        res                 -- int or (int, int) tuple giving the desired output
                                resolution of images
        """
        res = (res, res) if isinstance(res, int) else res

        image_class = os.path.basename(os.path.dirname(image))
        image_name = os.path.basename(image)
        unchanged_image = f"{source}/{image_class}/{image_name}"
        fixed_image = f"{fixed_image_dir}/{image_class}/{image_name}"
        if not os.path.exists(os.path.dirname(fixed_image)):
            os.makedirs(os.path.dirname(fixed_image))

        if len(np.array(Image.open(image)).shape) == 2:
            _ = Image.open(image).convert("RGB").save(fixed_image)
            tqdm.write(f"\t\tLOG: Image {unchanged_image} had no color channel. In the LMDB file, it will be made RGB.")
            image = fixed_image

        if (not res is None
            and not np.array(Image.open(image)).shape[:-1] == res):
            _ = Image.open(image).resize((res[0], res[1]), Image.BICUBIC).save(fixed_image)
            tqdm.write(f"\t\tLOG: Image {unchanged_image} had incorrect size {dims}. In the LMDB file, will be resized to spatial dimensions {res}.")
            image = fixed_image

        return image

    tqdm.write(f"\tLOG: LMDBification of {source}")

    ############################################################################
    # Find all images to LMDBify, check that they are the same resolution,
    # optionally fix them so they on't have common issues that'd prevent
    # LMDBification, and get the sum of their sizes (in bytes). Any fixes made
    # impact the LMDBified result and not the original image.
    ############################################################################
    images = [f"{source}/{cls_dir}/{image}"
        for cls_dir in os.listdir(source)
        for image in os.listdir(f"{source}/{cls_dir}")]
    images = {image for image in images if is_image(image)}
    if fix_images:
        fixed_image_dir = f"{os.path.dirname(__file__)}/fixed_images"
        images = {fix_image(image, fixed_image_dir) for image in tqdm(images,
            desc="Fixing potential issues in images",
            leave=False,
            dynamic_ncols=True)}

    image2dims = {image: np.array(Image.open(image)).shape for image in images}

    dimensions = list(image2dims.values())
    if not all([d == dimensions[0] for d in dimensions]):
        raise ValueError(f"All images must be the same size, but got sizes of {set(dimensions)}")
    num_bytes = np.sum([np.array(Image.open(image)).nbytes for image in images])


    ############################################################################
    # Create the LMDB file.
    ############################################################################
    tqdm.write(f"\tLOG: Will write output LMDB file to {out_path}")
    env = lmdb.open(out_path, map_size=num_bytes * 10)
    keys = [_write_lmdb_img(env, image) for image in tqdm(images,
        desc="Writing to LMDB file",
        leave=False,
        dynamic_ncols=True)]

    keys = ",".join(keys)
    with env.begin(write=True) as txn:
        txn.put("keys".encode("ascii"), keys.encode("ascii"))

    if fix_images:
        shutil.rmtree(fixed_image_dir)
    tqdm.write("\tDone.\n")

def all_image_folders_to_lmdb(source, replacing_path=None):
    """Writes every folder that can be read into an ImageFolder inside [source]
    (possibly nested many levels into the file hierarchy) to an LMDB file, while
    otherwise preserving the file hierarchy of [source].

    Args:
    source  -- directory in which to create LMDB files for constituent
                ImageFolder-interpretable directories
    enclosing_dir -- output directory in which to place results. This directory will
                have the same file structure as [source] internally
    """

    def is_image_folder(f):
        """Returns if folder [f] can be interpreted as an ImageFolder."""
        dirs = [d for d in os.listdir(f) if os.path.isdir(f"{f}/{d}")]
        return (len(dirs) > 0 and all(
                [all([is_image(f"{d}/{sd}") for sd in os.listdir(f"{f}/{d}")])
                for d in dirs]))

    def get_directory_hierarchy_flat(f):
        """Returns a nested list giving the file hierarchy under folder [f]."""
        result = []
        if not os.path.isdir(f):
            pass
        elif not any([os.path.isdir(f"{f}/{d}") for d in os.listdir(f)]):
            result += [f]
        else:
            for d in os.listdir(f):
                result += get_directory_hierarchy_flat(f"{f}/{d}")
                result += [f"{f}/{d}"] if os.path.isdir(f"{f}/{d}") else []
        return result

    replacing_path = os.path.dirname(source) if replacing_path is None else replacing_path
    folders = get_directory_hierarchy_flat(source) + [source]
    folders = [f for f in folders if is_image_folder(f)]
    tqdm.write(f"LOG: Searching over possible ImageFolder directories {sorted(folders)}")

    for f in tqdm(sorted(folders, reverse=True),
        desc="Writing ImageFolders to LMDB files - Inner loop",
        leave=False,
        dynamic_ncols=True):

        out_path = f.replace(source, replacing_path)
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))

        image_folder_to_lmdb(f, f"{out_path}.lmdb")

    tqdm.write("-------------------------------------------")


def merge_lmdb_files(lmdb_file_list, out_path, tmp_image_dir=f"./tmp_images"):
    """Concatenates the files in [lmdb_file_list] into one at [out_oath].
    [lmdb_file_list] can be a

    """
    if isinstance(lmdb_file_list, (list, tuple, set)):
        pass
    elif isinstance(lmdb_file_list, str) and lmdb_file_list.endswith("*"):
        lmdb_file_list = glob.glob(lmdb_file_list)
    else:
        raise ValueError()

    num_bytes = 0
    all_keys = []
    for lmdb_file in lmdb_file_list:
        env = lmdb.open(lmdb_file, readonly=True, lock=False, readahead=False,
            meminit=False)
        with env.begin(write=False) as txn:
            keys = txn.get("keys".encode("ascii"))
            keys = sorted([k for k in keys.decode("ascii").split(",")])
        num_bytes += np.sum([np.array(_read_lmdb_img(env, k), dtype=np.uint8).nbytes
            for k in keys])
        all_keys += keys

    if not os.path.exists(tmp_image_dir):
        os.makedirs(tmp_image_dir)
    out_env = lmdb.open(out_path, map_size=num_bytes * 10)
    for lmdb_file in tqdm(lmdb_file_list,
        desc="Reading LMDB files",
        dynamic_ncols=True):

        read_env = lmdb.open(lmdb_file, readonly=True, lock=False,
            readahead=False, meminit=False)
        with read_env.begin(write=False) as txn:
            keys = txn.get("keys".encode("ascii"))
            keys = sorted([k for k in keys.decode("ascii").split(",")])
        for k in keys:
            image_dir = os.path.dirname(k)
            if not os.path.exists(f"{tmp_image_dir}/{image_dir}"):
                os.makedirs(f"{tmp_image_dir}/{image_dir}")
            image_file = f"{tmp_image_dir}/{k}"
            _ = _read_lmdb_img(read_env, k).save(image_file)
            _write_lmdb_img(out_env, image_file)
            os.remove(image_file)

    all_keys = ",".join(all_keys)
    with out_env.begin(write=True) as txn:
        txn.put("keys".encode("ascii"), all_keys.encode("ascii"))

    shutil.rmtree(tmp_image_dir)

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--data_dirs", required=True, nargs="*",
        help="Path under and including which to search for ImageFolder-compatible folders to LMDBify.")
    P.add_argument("--enclosing_dir", default=None,
        help="Enclosing directory where all LMDB datasets go")
    P.add_argument("--new_name",
        help="Name of the LMDB file. Defaults to --data_dir.")
    P.add_argument("--ignore_existing_lmdb", choices=[0, 1], default=1,
        help="Ignore existing LMDB files")
    args = P.parse_args()

    ############################################################################
    # Check/modify args
    ############################################################################
    data_dirs = [os.path.abspath(d.rstrip("/")) for d in args.data_dirs if os.path.isdir(d)]

    if args.ignore_existing_lmdb:
        data_dirs = [d for d in data_dirs if not "lmdb" in d.lower()]

    data_dir_basenames = [os.path.basename(d) for d in data_dirs]
    cmn_sbstr = longest_common_starting_substring(data_dir_basenames).strip("_")
    args.new_name = cmn_sbstr if args.new_name is None else args.new_name
    new_names = [f"{os.path.dirname(d)}/{os.path.basename(d).replace(cmn_sbstr, args.new_name)}"
        for d in data_dirs]

    args.data_dirs = data_dirs
    args.new_names = new_names
    args.enclosing_dir = os.path.abspath(args.enclosing_dir.rstrip("/"))

    ############################################################################
    # Log the conversions that will happen
    ############################################################################
    data_dir2new_name = {d: n for d,n in zip(args.data_dirs, args.new_names)}
    conversion_strs = [f"{d:60} ->    {args.enclosing_dir}/{n.replace(os.path.dirname(n), '').strip('/'):40}" for d,n in data_dir2new_name.items()]
    conversion_str = "\n\t".join(sorted(conversion_strs))
    tqdm.write(f"LOG: Will attempt the following LMDB conversions:\n\n\t{conversion_str}\n")

    for d,n in tqdm(list(zip(args.data_dirs, args.new_names)),
        desc="Writing ImageFolders to LMDB files - Outer loop",
        leave=True,
        dynamic_ncols=True):

        all_image_folders_to_lmdb(d, replacing_path=n)
