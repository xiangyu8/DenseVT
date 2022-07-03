import torch
import cv2
from turbojpeg import TurboJPEG
import numpy as np

from . import cvfunctional as F

from .dctparameters import subset_channel_index
from PIL import ImageEnhance, Image
import random
import math

__all__ = ["ToTensorDCT", "NormalizeDCT", "Upscale", "Average", 
            "AdjustDCT", "DCTCenterCrop", "SubsetDCT", "ToCVImage","ToImageCV" 
           "Aggregate","GetDCT_resize","GetDCT","UpScaleDCT","Shift" 
           ]


class ToImageCV(object):
    def __call__(self, img):
        return np.asarray(img)

class Upscale(object):
    def __init__(self, upscale_factor=2, interpolation='BILINEAR'):
        self.upscale_factor = upscale_factor
        self.interpolation = interpolation

    def __call__(self, img):
        return img, F.upscale(img, self.upscale_factor, self.interpolation)

class Permute(object):
    def __call__(self, img):
        index = torch.randperm(img.shape[0])
        out = img[index]
        return out

class Shift(object):
    def __call__(self,img):
        k = torch.randint(1, img.shape[0]//4, (1,)).item()
        out = torch.randn(img.shape)
        out[k:] = img[0:img.shape[0]-k]
        return out

class TransformUpscaledDCT(object):
    def __init__(self):
        self.jpeg_encoder = TurboJPEG()
        # self.jpeg_encoder = TurboJPEG('/home/kai.x/work/local/lib/libturbojpeg.so')

    def __call__(self, img):
        y, cbcr = img[0], img[1]
        dct_y, _, _ = F.transform_dct(y, self.jpeg_encoder)
        _, dct_cb, dct_cr = F.transform_dct(cbcr, self.jpeg_encoder)
        return dct_y, dct_cb, dct_cr

class UpScaleDCT(object):
    # upscale cb and cr channels since they are having half of the size of y channel.
    def __call__(self,img):
        #print("shape before upscaling: ", img[0].shape,img[1].shape,img[2].shape)
        y,cb,cr = img[0],img[1],img[2]
        size = y.shape[-2]
        cb = cv2.resize(cb,(size,size))
        cr = cv2.resize(cr,(size,size))
        #print("inside of UpscaleDCT: shape of img ",y.shape)
        return y,cb,cr

class GetDCT_resize(object):
    def __init__(self, dct_filter_size = 8):
        self.jpeg_encoder = TurboJPEG()
        self.dct_filter_size = dct_filter_size
        self.interpolation = "BILINEAR"

    def __call__(self,img):
      #  img = F.resize(img, self.dct_filter_size*img.shape[-2]//2, self.interpolation)   
        if self.dct_filter_size ==8:
            dct_y, dct_cb, dct_cr = F.transform_dct(img, self.jpeg_encoder)
        else:
            raise ValueError ("Only 8x8 is supported here.")
            dct_y, dct_cb, dct_cr = F.transform_dct_size(img, self.jpeg_encoder, self.dct_filter_size)
        return dct_y, dct_cb, dct_cr


class RandomHorizontalFlip(object):
    """Horizontally flip the given CV Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (CV Image): Image to be flipped.

        Returns:
            CV Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class Resize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``BILINEAR``
    """

    def __init__(self, size, interpolation='BILINEAR'):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (np.ndarray): Image to be scaled.

        Returns:
            np.ndarray: Rescaled image.
        """
        return F.resize(img, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = self.interpolation
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)
class CenterCrop(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):

        self.size = size

    def __call__(self, img):
        """
        Args:
            img (CV Image): Image to be cropped.

        Returns:
            CV Image: Cropped image.
        """
        return F.center_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class RandomResizedCrop(object):
    """Crop the given CV Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='BILINEAR'):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (CV Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if random.random() < 0.5:
                w, h = h, w

            if w <= img.shape[1] and h <= img.shape[0]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return i, j, h, w

        # Fallback
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return i, j, w, w
    def __call__(self, img):
        """
        Args:
            img (np.ndarray): Image to be cropped and resized.

        Returns:
            np.ndarray: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = self.interpolation
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string

transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)
class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        out = np.array(out)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return out

class GetDCT(object):
    def __init__(self, dct_filter_size = 8):
        self.jpeg_encoder = TurboJPEG()
        self.dct_filter_size = dct_filter_size
        self.interpolation = "BILINEAR"

    def __call__(self,img):
        #img = F.resize(img, self.dct_filter_size*56, self.interpolation)   
        if self.dct_filter_size ==8:
            dct_y, dct_cb, dct_cr = F.transform_dct(img, self.jpeg_encoder)
        else:
            raise ValueError ("Only 8x8 is supported here.")
            dct_y, dct_cb, dct_cr = F.transform_dct_size(img, self.jpeg_encoder, self.dct_filter_size)
        return dct_y, dct_cb, dct_cr

class ToTensorDCT(object):
    """Convert a ``numpy.ndarray`` to tensor.

    Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, img):
        """
        Args:
            pic (numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        y, cb, cr = img[0], img[1], img[2]
        y, cb, cr = F.to_tensor_dct(y), F.to_tensor_dct(cb), F.to_tensor_dct(cr)

        return y, cb, cr

class SubsetDCT(object):
    def __init__(self, channels=24):
        self.subset_channel_index = subset_channel_index
        if channels in [14,15, 16,17, 28, 36]:
            self.subset_y =  self.subset_channel_index[str(channels)][0]
        elif channels in [35]:
            self.subset_y =  self.subset_channel_index[str(channels)][0]
            self.subset_cb = self.subset_channel_index[str(channels)][1]
        elif channels not in [192, 768,108, 432, 300]:
            self.subset_y =  self.subset_channel_index[str(channels)][0]
            self.subset_cb = self.subset_channel_index[str(channels)][1]
            self.subset_cr = self.subset_channel_index[str(channels)][2]
        self.channels = channels

    def __call__(self, tensor):
        dct_y, dct_cb, dct_cr = tensor[0], tensor[1], tensor[2]
        if self.channels  in [14, 15,16,17, 28, 36]:
            dct_y = dct_y[self.subset_y]
            dct_cb, dct_cr = None, None
        elif self.channels  in [35]:
            dct_y, dct_cb = dct_y[self.subset_y],dct_cb[self.subset_cb]
            dct_cr = None
        elif self.channels not in [192, 768, 108, 432, 300]:
            dct_y, dct_cb, dct_cr = dct_y[self.subset_y], dct_cb[self.subset_cb], dct_cr[self.subset_cr]

        #print("inside subsetDct:")
        #print("shape of tensor: ", tensor[0].shape)
        #print("shape of dct_y: ", dct_y.shape)
        return dct_y, dct_cb, dct_cr

class Aggregate(object):
    def __call__(self, img):
       # print("Inside Aggregate:")
       # print("shape of img: ",img[0].shape, img[1].shape,img[2].shape)
        dct_y, dct_cb, dct_cr = img[0], img[1], img[2]
        if dct_cb is not None:
            if dct_cr is not None:
                dct_y = torch.cat((dct_y, dct_cb, dct_cr), dim=0)
            else:
                dct_y = torch.cat((dct_y, dct_cb), dim=0)
        return dct_y

class NormalizeDCT(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """
    def __init__(self, y_mean, y_std, cb_mean=None, cb_std=None, cr_mean=None, cr_std=None, channels=None):
        self.y_mean,  self.y_std = y_mean, y_std
        self.cb_mean, self.cb_std = cb_mean, cb_std
        self.cr_mean, self.cr_std = cr_mean, cr_std

        if channels is None or (channels in [192, 768, 108, 432, 300]):
            self.mean_y, self.std_y = y_mean, y_std
            
        else:
            self.subset_channel_index = subset_channel_index
            self.subset_y  = self.subset_channel_index[str(channels)][0]
            self.subset_cb = self.subset_channel_index[str(channels)][1]
            self.subset_cb = [64+c for c in self.subset_cb]
            self.subset_cr = self.subset_channel_index[str(channels)][2]
            self.subset_cr = [128+c for c in self.subset_cr]
            self.subset = self.subset_y + self.subset_cb + self.subset_cr
            self.mean_y, self.std_y = [y_mean[i] for i in self.subset], [y_std[i] for i in self.subset]

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if isinstance(tensor, list):
            y, cb, cr = tensor[0], tensor[1], tensor[2]
            y  = F.normalize(y,  self.y_mean,  self.y_std)
            cb = F.normalize(cb, self.cb_mean, self.cb_std)
            cr = F.normalize(cr, self.cr_mean, self.cr_std)
            #print("exiting normalized!")
            return y, cb, cr
        else:
            y = F.normalize(tensor, self.mean_y, self.std_y)
            #print("type of final transformation: ", type(y))
            #print("exiting normalized!")  
#            return y, None, None
            return y

class DCTCenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        y, cb, cr = img[0], img[1], img[2]
        y = F.center_crop(y, self.size)
        cb = F.center_crop(cb, self.size)
        cr = F.center_crop(cr, self.size)

        return y, cb, cr

class Average(object):
    def __call__(self, img):
        if isinstance(img, list):
            y, cb, cr = img[0], img[1], img[2]
            y = y.view(y.size(0), -1).mean(dim=1)
            cb = cb.view(cb.size(0), -1).mean(dim=1)
            cr = cr.view(cr.size(0), -1).mean(dim=1)
            return y, cb, cr
        else:
            return img.view(img.size(0), -1).mean(dim=1), None, None

class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Attention: The multiprocessing used in dataloader of pytorch is not friendly with lambda function in Windows

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        # assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd
        # if 'Windows' in platform.system():
        #     raise RuntimeError("Can't pickle lambda funciton in windows system")

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToTensor(object):
    """Convert a ``numpy.ndarray`` to tensor.

    Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        """
        Args:
            pic (numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToCVImage(object):
    """Convert a tensor or an to ndarray Image.

    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a CV Image while preserving the value range.

    Args:
        mode (str): color space and pixel depth of input data (optional).
    """
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, pic):
        """
        Args:
            pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.

        Returns:
            PIL Image: Image converted to PIL Image.

        """
        return F.to_cv_image(pic, self.mode)



