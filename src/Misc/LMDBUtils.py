"""Code for working with LMDB files.

--------------------------------------------------------------------------------
The files have the following properties:
(1) They are assumed to store a key-value mapping, with a special key 'keys'
    giving a string of all the other keys, separated by commas
(2) Each represents a folder compatible with TorchVision's ImageFolder class,
    ie. keys are of the form 'class/file'.

--------------------------------------------------------------------------------
The functions herein in general have the following properties:
(1) They are agnostic to whether passed-in LMDB "files" are opened LMDB
    environments or simply the paths to the LMDB files themselves.
"""
import argparse
import numpy as np
import os
from tqdm import tqdm
import lmdb
from torch.utils.data import Dataset
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor
from PIL import Image
import shutil
import glob
import torch

import random
import string

################################################################################
# Miscellaneous utility functions.
################################################################################

def is_dir(f):
    try:
        _ = os.listdir(f)
        return True
    except NotADirectoryError as e:
        return False

    raise ValueError()


def is_image_folder(f):
    """Returns if folder [f] can be interpreted as an ImageFolder."""
    if is_dir(f):
        dirs = [d for d in os.listdir(f) if is_dir(f"{f}/{d}")]
        return (len(dirs) > 0 and all(
                [all([is_image(f"{d}/{sd}") for sd in os.listdir(f"{f}/{d}")])
                for d in dirs]))
    else:
        return False

def lmdb_file_contains_augs(lmdb_file):
    """Returns if LMDB file [lmdb_file] contains images that should be
    interpreted as augmentations rather than images in their own right.
    """
    return any(["_aug" in k for k in lmdb_to_keys(lmdb_file)])

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

def has_image_extension(x):
    """Returns if [x] ends with an image file extension."""
    x = x.lower()
    return x.endswith(".jpg") or x.endswith(".png") or x.endswith(".jpeg")

def get_random_string(length=16):
    """Returns a random ASCII string of length [length]."""
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))

def get_temporary_storage_folder(length=16):
    """Returns and creates a folder for temporary storage. Functions using this
    should delete the folder prior to returning. Normally, this would just be
    the /tmp directory, but this isn't available on ComputeCanada.
    """
    rand_str = get_random_string(length)
    folder = f"{os.path.dirname(__file__)}/tmp_storage_{rand_str}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def copy_lmdb_into_lmdb(x, y, store_image=True):
    """Returns LMDB environment [y] after copying the contents of LMDB
    environment [x] into it. [store_image] should be set to indicate whether the
    copied data is an image or not.

    This is relatively slow, but the API is reasonable.
    """
    tmp_dir = get_temporary_storage_folder()

    keys = lmdb_to_keys(x)
    for k in tqdm(keys,
        leave=False,
        desc="Copying contents of one LMDB to another",
        dynamic_ncols=True):

        write_to_lmdb(y, k,
            value=read_image_from_lmdb(x, k),
            store_image=store_image,
            tmp_dir=tmp_dir)

    shutil.rmtree(tmp_dir)
    return y

def lmdb_to_keys(lmdb_file, images_only=True):
    """Returns the list of keys in [lmdb_file]. [lmdb_file] can be either the
    path to an LMDB file, or an LMDB file opened for reading.
    """
    if isinstance(lmdb_file, str) and lmdb_file.endswith(".lmdb"):
        env = lmdb.open(lmdb_file, readonly=True, lock=False, readahead=False,
            meminit=False)
    elif isinstance(lmdb_file, lmdb.Environment):
        env = lmdb_file
    else:
        raise ValueError()

    with env.begin(write=False) as txn:
        keys = [k.decode("ascii") for k in tqdm(txn.cursor().iternext(keys=True, values=False), leave=False, desc="Getting LMDB keys")]
        return [k for k in keys if not k.endswith(".dims") and not k == "keys"]

################################################################################
# Core functions that are part of the API.
################################################################################
def write_to_lmdb(lmdb_file, key, value=None, store_image=True, tmp_dir=None):
    """Writes the key-value pair ([key], [value]) to [lmdb_file]. If [lmdb_file]
    does not exist, it is created, and resizing if necessary is supported.

    WARNING: This function isn't thread safe or parallelizable, but it's a
    reasonable LMDB API.

    Args:
    lmdb_file   -- file or environment to write [key] and [value] to
    key         -- the key to write, with form 'class/name'
    value       -- the value to write to [key]. If [store_image] is specified,
                    then this can be any of (a) a path to an image, (b) a PIL
                    image, (c) a CxHxW tensor, (d) None, in which [key] is
                    assumed to be a path to the image. Otherwise, it will be
                    encoded as ASCII and stored directly.
    store_image -- whether to interpret [value] as an image
    tmp_dir     -- directory to store images during writing. If None, it will be
                    created and removed while this function executes. Otherwise,
                    it will be used, files in it may be overwritten, and it
                    won't be deleted. This argument should be specified whenever
                    an existing temporary directory is already in use.
    """
    def _write_lmdb_img(env, image, key=None):
        """Writes the image stored at file [image] to LMDB dataset open for
        writing [env] with key [key] if [key] isn't None, or otherwise the name
        and parent folder of [image].
        """
        img_name = os.path.basename(image)
        img_dir = os.path.dirname(image)
        key = f"{os.path.basename(img_dir)}/{img_name}" if key is None else key

        image = np.array(Image.open(image), dtype=np.uint8)
        dims = " ".join([f"{d:d}" for d in image.shape])
        with env.begin(write=True) as txn:
            txn.put(key.encode("ascii"), image)
            txn.put(f"{key}.dims".encode("ascii"), dims.encode("ascii"))

        keys = set(lmdb_to_keys(env) + [key])
        keys = ",".join(keys)
        with env.begin(write=True) as txn:
            txn.put("keys".encode("ascii"), keys.encode("ascii"))

    ############################################################################
    # Get and create the folder to which we'll temporarily store the images
    # during copying, and the fixed path we'll use for this.
    ############################################################################
    should_erase_tmp_dir = (tmp_dir is None)
    tmp_dir = get_temporary_storage_folder() if tmp_dir is None else tmp_dir
    tmp_image = f"{tmp_dir}/image.jpg"

    ############################################################################
    # Move the image we want to store to [tmp_image] and optionally modify [key]
    # if needed.
    ############################################################################
    if (store_image and value is None and os.path.exists(key)
        and has_image_extension(key)):
        # In this case, [key] is a path to the image to store. We need to copy
        # it to the path whose contents will be written into the LMDB file, and
        # extract the proper key to use for this.
        shutil.copyfile(key, tmp_image)
        key = f"{os.path.basename(os.path.dirname(key))}/{os.path.basename(key)}"
    elif store_image and isinstance(value, str):
        shutil.copyfile(value, tmp_image)
    elif store_image and isinstance(value, Image.Image):
        save_image(to_tensor(value), tmp_image)
    elif (store_image and isinstance(value, torch.Tensor)
        and len(value.shape) == 3 and value.shape[0] == 3):
        save_image(value, tmp_image)
    elif (store_image and isinstance(value, torch.Tensor)
        and len(value.shape) == 4 and value.shape[0] == 1
        and value.shape[1] == 3):
        save_image(value.squeeze(0), tmp_image)
    elif not store_image:
        raise NotImplementedError()
    else:
        raise ValueError(f"Could not match image: {tmp_image} | store_image {store_image} | value {value} | {key} | {os.path.exists(key)}")

    ############################################################################
    # Open the LMDB file we want to write the image to.
    ############################################################################
    if isinstance(lmdb_file, str) and os.path.exists(lmdb_file):
        env = lmdb.open(lmdb_file)
    elif isinstance(lmdb_file, str) and not os.path.exists(lmdb_file):
        env = lmdb.open(lmdb_file, map_size=8192)
    elif isinstance(lmdb_file, lmdb.Environment):
        env = lmdb_file
    else:
        raise ValueError(f"Unmatched case")

    ############################################################################
    # Try and copy the contents of [tmp_image] to the opened LMDB file. If this
    # fails due to the LMDB file not being big enough, try again after enlarging
    # the LMDB file.
    ############################################################################
    try:
        _write_lmdb_img(env, tmp_image, key=key)
    except lmdb.MapFullError as e:
        env.set_mapsize(env.info()["map_size"] * 10)
        write_to_lmdb(env, key, value)

    if should_erase_tmp_dir:
        shutil.rmtree(tmp_dir)


def read_image_from_lmdb(lmdb_file, key):
    """Returns a PIL image from [key] in opened LMDB file [lmdb_file]."""
    if isinstance(lmdb_file, str) and os.path.exists(lmdb_file):
        env = lmdb.open(lmdb_file)
    elif isinstance(lmdb_file, lmdb.Environment):
        env = lmdb_file
    else:
        raise ValueError(f"Unmatched case")

    with env.begin(write=False) as txn:
        buf = txn.get(key.encode("ascii"))
        buf_dims = txn.get(f"{key}.dims".encode("ascii")).decode("ascii")
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    h, w, c = [int(s) for s in buf_dims.split()]
    img = img_flat.reshape(h, w, c)
    return Image.fromarray(img)

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

    def loader(self, key):
        """Returns a PIL image from [self] given its key."""
        with self.env.begin(write=False) as txn:
            buf = txn.get(key.encode("ascii"))
            buf_dims = txn.get(f"{key}.dims".encode("ascii")).decode("ascii")
        img_flat = np.frombuffer(buf, dtype=np.uint8)
        h, w, c = [int(s) for s in buf_dims.split()]
        x = img_flat.reshape(h, w, c)
        return Image.fromarray(x)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        assert idx == int(idx), idx
        idx = int(idx)
        key, y = self.samples[idx]

        # Function inlining to increase speed. Equivalent to read_image_from_lmdb().
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
    
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(outpath))

    env = lmdb.open(out_path, map_size=num_bytes * 10)
    for image in tqdm(images,
        desc="Writing to LMDB file",
        leave=False,
        dynamic_ncols=True):
        write_to_lmdb(env, image)

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
    def get_directory_hierarchy_flat(f):
        """Returns a nested list giving the file hierarchy under folder [f]."""
        result = []
        if not is_dir(f):
            pass
        elif not any([is_dir(f"{f}/{d}") for d in os.listdir(f)]):
            result += [f]
        else:
            for d in os.listdir(f):
                result += get_directory_hierarchy_flat(f"{f}/{d}")
                result += [f"{f}/{d}"] if is_dir(f"{f}/{d}") else []
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


def merge_lmdb_files(lmdb_file_list, out_path):
    """Concatenates the files in [lmdb_file_list] into one at [out_oath].
    [lmdb_file_list] can be a

    """
    if isinstance(lmdb_file_list, (list, tuple, set)):
        pass
    elif isinstance(lmdb_file_list, str) and lmdb_file_list.endswith("*"):
        lmdb_file_list = glob.glob(lmdb_file_list)
    else:
        raise ValueError()

    for lmdb_file in tqdm(lmdb_file_list,
        desc="Reading LMDB files",
        dynamic_ncols=True):

        read_env = lmdb.open(lmdb_file, readonly=True, lock=False,
            readahead=False, meminit=False)
        _ = copy_lmdb_into_lmdb(read_env, out_path)

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
    data_dirs = [os.path.abspath(d.rstrip("/")) for d in args.data_dirs if is_dir(d)]

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
