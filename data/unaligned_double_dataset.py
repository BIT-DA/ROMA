import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
import torchvision.transforms.functional as TF
import random
from torchvision.transforms import transforms as tfs

class UnalignedDoubleDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # self.use_resize_crop = opt.use_resize_crop
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.opt = opt
        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        A0 = A_img.crop((0,0,256,256))
        A1 = A_img.crop((256,0,512,256))
        B_img = Image.open(B_path).convert('RGB')
        B0 = B_img.crop((0,0,256,256))
        B1 = B_img.crop((256,0,512,256))

        # Apply image transformation
        # For FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
#        print('current_epoch', self.current_epoch)
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)

        resize = tfs.Resize(size=(self.opt.load_size, self.opt.load_size))
        imgA = resize(A0)
        param = dict()
        i, j, h, w = tfs.RandomCrop.get_params(
            imgA, output_size=(self.opt.crop_size, self.opt.crop_size))
        param['crop_pos'] = (i, j)
        transform = get_transform(modified_opt, param)
        # print(transform)
        # sys.exit(0)
        # A = transform(A_img)
        # B = transform(B_img)

        A0 = transform(A0)
        B0 = transform(B0)
        A1 = transform(A1)
        B1 = transform(B1)

        return {'A0': A0, 'A1': A1, 'B0': B0, 'B1': B1, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
