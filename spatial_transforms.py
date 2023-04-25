import random
import numbers
import collections
import warnings
from typing import Tuple, Optional, List, Sequence

import math
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
import scipy.ndimage
from PIL import Image
from torch import Tensor
import cv2
import torchvision.transforms as transforms

try:
    import accimage
except ImportError:
    accimage = None


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """

        if isinstance(pic, np.ndarray):
            # handle numpy array
            # pic = pic.astype(np.float64)
            img = torch.from_numpy(np.expand_dims(pic.copy(), axis=0))
            # backward compatibility
            return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic.copy())

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

    def randomize_parameters(self):
        pass


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

    def randomize_parameters(self):
        pass


class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size,
                          int) or (isinstance(size, collections.Iterable) and
                                   len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

    def randomize_parameters(self):
        pass


class CenterCrop(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))

    def randomize_parameters(self):
        pass


class CornerCrop(object):

    def __init__(self, size, crop_position=None):
        self.size = size
        if crop_position is None:
            self.randomize = True
        else:
            self.randomize = False
        self.crop_position = crop_position
        self.crop_positions = ['c', 'tl', 'tr', 'bl', 'br']

    def __call__(self, img):
        image_width = img.size[0]
        image_height = img.size[1]

        if self.crop_position == 'c':
            th, tw = (self.size, self.size)
            x1 = int(round((image_width - tw) / 2.))
            y1 = int(round((image_height - th) / 2.))
            x2 = x1 + tw
            y2 = y1 + th
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = self.size
            y2 = self.size
        elif self.crop_position == 'tr':
            x1 = image_width - self.size
            y1 = 0
            x2 = image_width
            y2 = self.size
        elif self.crop_position == 'bl':
            x1 = 0
            y1 = image_height - self.size
            x2 = self.size
            y2 = image_height
        elif self.crop_position == 'br':
            x1 = image_width - self.size
            y1 = image_height - self.size
            x2 = image_width
            y2 = image_height

        img = img.crop((x1, y1, x2, y2))

        return img

    def randomize_parameters(self):
        if self.randomize:
            self.crop_position = self.crop_positions[random.randint(
                0,
                len(self.crop_positions) - 1)]


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.p < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def randomize_parameters(self):
        # random.seed(1337)
        self.p = random.random()


class MultiScaleCornerCrop(object):
    """Crop the given PIL.Image to randomly selected size.
    A crop of size is selected from scales of the original size.
    A position of cropping is randomly selected from 4 corners and 1 center.
    This crop is finally resized to given size.
    Args:
        scales: cropping scales of the original size
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self,
                 scales,
                 size,
                 interpolation=Image.BILINEAR,
                 crop_positions=['c', 'tl', 'tr', 'bl', 'br']):
        self.scales = scales
        self.size = size
        self.interpolation = interpolation

        self.crop_positions = crop_positions

    def __call__(self, img):
        min_length = min(img.size[0], img.size[1])
        crop_size = int(min_length * self.scale)

        image_width = img.size[0]
        image_height = img.size[1]

        if self.crop_position == 'c':
            center_x = image_width // 2
            center_y = image_height // 2
            box_half = crop_size // 2
            x1 = center_x - box_half
            y1 = center_y - box_half
            x2 = center_x + box_half
            y2 = center_y + box_half
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = crop_size
            y2 = crop_size
        elif self.crop_position == 'tr':
            x1 = image_width - crop_size
            y1 = 0
            x2 = image_width
            y2 = crop_size
        elif self.crop_position == 'bl':
            x1 = 0
            y1 = image_height - crop_size
            x2 = crop_size
            y2 = image_height
        elif self.crop_position == 'br':
            x1 = image_width - crop_size
            y1 = image_height - crop_size
            x2 = image_width
            y2 = image_height

        img = img.crop((x1, y1, x2, y2))

        return img.resize((self.size, self.size), self.interpolation)

    def randomize_parameters(self):
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        self.crop_position = self.crop_positions[random.randint(
            0,
            len(self.scales) - 1)]


class MultiScaleRandomCrop(object):

    def __init__(self, scales, size, interpolation=Image.BILINEAR):
        self.scales = scales
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        min_length = min(img.size[0], img.size[1])
        crop_size = int(min_length * self.scale)

        image_width = img.size[0]
        image_height = img.size[1]

        x1 = self.tl_x * (image_width - crop_size)
        y1 = self.tl_y * (image_height - crop_size)
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        img = img.crop((x1, y1, x2, y2))

        return img.resize((self.size, self.size), self.interpolation)

    def randomize_parameters(self):
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        # self.scale = 1
        self.tl_x = random.random()
        self.tl_y = random.random()


class Hog_descriptor(object):
    def __init__(self, cell_size=8, bin_size=8):

        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 // self.bin_size
        assert type(self.bin_size) == int, "bin_size should be integer,"
        assert type(self.cell_size) == int, "cell_size should be integer,"
        assert type(self.angle_unit) == int, "bin_size should be divisible by 360"

    def global_gradient(self):
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return gradient_magnitude, gradient_angle

    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        if idx == self.bin_size:
            return idx - 1, (idx) % self.bin_size, mod
        return idx, (idx + 1) % self.bin_size, mod

    def render_gradient(self, image, cell_gradient):
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image

    def __call__(self, img):
        img = np.array(img)
        img = np.squeeze(img)
        self.img = np.squeeze(img)
        self.img = np.sqrt(img / float(np.max(img)))
        self.img = self.img * 255
        height, width = self.img.shape
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)
        cell_gradient_vector = np.zeros((height // self.cell_size, width // self.cell_size, self.bin_size))
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)
        hog_vector = []
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.append(block_vector)
        return Image.fromarray(hog_image)

    def randomize_parameters(self):
        pass


class SpatialElasticDisplacement(object):

    def __init__(self, sigma=3.0, alpha=1.0, order=3, cval=0, mode="constant"):
        self.alpha = alpha
        self.sigma = sigma
        self.order = order
        self.cval = cval
        self.mode = mode

    def __call__(self, img):
        if self.p < 0.65:
            is_PIL = isinstance(img, Image.Image)
            if is_PIL:
                img = np.asarray(img)

            image = img
            image_first_channel = np.squeeze(image[..., 0])
            indices_x, indices_y = self._generate_indices(image_first_channel.shape, alpha=self.alpha, sigma=self.sigma)
            ret_image = (self._map_coordinates(
                image,
                indices_x,
                indices_y,
                order=self.order,
                cval=self.cval,
                mode=self.mode))

            if is_PIL:
                return Image.fromarray(ret_image)
            else:
                return ret_image
        else:
            return img

    def _generate_indices(self, shape, alpha, sigma):
        assert (len(shape) == 2), "shape: Should be of size 2!"
        dx = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        return np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    def _map_coordinates(self, image, indices_x, indices_y, order=1, cval=0, mode="constant"):
        assert (len(image.shape) == 3), "image.shape: Should be of size 3!"
        result = np.copy(image)
        height, width = image.shape[0:2]
        for c in range(image.shape[2]):
            remapped_flat = scipy.ndimage.interpolation.map_coordinates(
                image[..., c],
                (indices_x, indices_y),
                order=order,
                cval=cval,
                mode=mode
            )
            remapped = remapped_flat.reshape((height, width))
            result[..., c] = remapped
        return result

    def randomize_parameters(self):
        self.p = random.random()


class RandomRotate(object):

    def __init__(self):
        self.interpolation = Image.BILINEAR

    def __call__(self, img):
        im_size = img.size
        ret_img = img.rotate(self.rotate_angle, resample=self.interpolation)

        return ret_img

    def randomize_parameters(self):
        # random.seed(1337)
        self.rotate_angle = random.randint(-5, 5)


class RandomResize(object):

    def __init__(self):
        self.interpolation = Image.BILINEAR

    def __call__(self, img):
        im_size = img.size
        ret_img = img.resize((int(im_size[0] * self.resize_const),
                              int(im_size[1] * self.resize_const)))
        return ret_img

    def randomize_parameters(self):
        self.resize_const = random.uniform(0.9, 1.1)


class Gaussian_blur(object):

    def __init__(self, radius=0.1):
        self.radius = radius

    def __call__(self, img):
        if self.p < 0.2:
            blurred = scipy.ndimage.gaussian_filter(img, sigma=5, order=0)
            return blurred
        else:
            return img

    def randomize_parameters(self):
        self.p = random.random()
        self.radius = random.uniform(0.0, self.radius)


class SaltImage(object):
    def __init__(self, ratio=100):
        self.ratio = ratio

    def __call__(self, img):
        is_PIL = isinstance(img, Image.Image)
        if is_PIL:
            img = np.asarray(img)

        if self.p < 0.10:
            data_final = []
            img = img.astype(np.float)
            img_shape = img.shape
            noise = np.random.randint(self.ratio, size=img_shape)
            img = np.where(noise == 0, 255, img)

            if is_PIL:
                return Image.fromarray(img.astype(np.uint8))
            else:
                return img
        else:
            return img

    def randomize_parameters(self):
        # random.seed(1337)
        self.p = random.random()
        self.ratio = random.randint(80, 120)


class Dropout(object):

    def __init__(self, ratio=100):
        self.ratio = ratio

    def __call__(self, img):
        is_PIL = isinstance(img, Image.Image)
        if is_PIL:
            img = np.asarray(img)

        if self.p < 0.10:
            data_final = []
            img = img.astype(np.float)
            img_shape = img.shape
            noise = np.random.randint(self.ratio, size=img_shape)
            img = np.where(noise == 0, 0, img)
            if is_PIL:
                return Image.fromarray(img.astype(np.uint8))
            else:
                return img
        else:
            return img

    def randomize_parameters(self):
        # random.seed(1337)
        self.p = random.random()
        self.ratio = random.randint(30, 50)


class MultiplyValues():

    def __init__(self, value=0.2, per_channel=False):
        self.value = value
        self.per_channel = per_channel

    def __call__(self, img):
        is_PIL = isinstance(img, Image.Image)
        if is_PIL:
            img = np.asarray(img)

        image = img.astype(np.float64)
        image *= self.sample
        image = np.where(image > 255, 255, image)
        image = np.where(image < 0, 0, image)
        image = image.astype(np.uint8)

        if is_PIL:
            return Image.fromarray(image)
        else:
            return image

    def randomize_parameters(self):
        self.sample = random.uniform(1.0 - self.value, 1.0 + self.value)


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=1.0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

        self.threshold = p

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, img):
        if self.p < self.threshold:
            return self.transform(img)
        return img

    def randomize_parameters(self):
        self.p = random.random()

        brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
        contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
        saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
        hue_factor = random.uniform(self.hue[0], self.hue[1])

        transforms = []
        transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))
        transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))
        transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))
        transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        self.transform = torchvision.transforms.Compose(transforms)


class RandomErasing(object):
    """Randomly selects a rectangle region in an torch Tensor image and erases its pixels.
    This transform does not support PIL Image.
    'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896

    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input image.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         inplace: boolean to make this transform inplace. Default set to False.

    Returns:
        Erased Image.

    Example:
        >>> transform = transforms.Compose([
        >>>   transforms.RandomHorizontalFlip(),
        >>>   transforms.PILToTensor(),
        >>>   transforms.ConvertImageDtype(torch.float),
        >>>   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>>   transforms.RandomErasing(),
        >>> ])
    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        super().__init__()
        if not isinstance(value, (numbers.Number, str, tuple, list)):
            raise TypeError("Argument value should be either a number or str or a sequence")
        if isinstance(value, str) and value != "random":
            raise ValueError("If value is str, it should be 'random'")
        if not isinstance(scale, (tuple, list)):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, (tuple, list)):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("Scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("Random erasing probability should be between 0 and 1")

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    @staticmethod
    def get_params(
            img: Tensor, scale: Tuple[float, float], ratio: Tuple[float, float], value: Optional[List[float]] = None
    ) -> Tuple[int, int, int, int, Tensor]:
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image to be erased.
            scale (sequence): range of proportion of erased area against input image.
            ratio (sequence): range of aspect ratio of erased area.
            value (list, optional): erasing value. If None, it is interpreted as "random"
                (erasing each pixel with random values). If ``len(value)`` is 1, it is interpreted as a number,
                i.e. ``value[0]``.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
        area = img_h * img_w

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < img_h and w < img_w):
                continue

            if value is None:
                v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
            else:
                v = torch.tensor(value)[:, None, None]

            i = torch.randint(0, img_h - h + 1, size=(1,)).item()
            j = torch.randint(0, img_w - w + 1, size=(1,)).item()
            return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        if torch.rand(1) < self.p:

            # cast self.value to script acceptable type
            if isinstance(self.value, (int, float)):
                value = [self.value]
            elif isinstance(self.value, str):
                value = None
            elif isinstance(self.value, tuple):
                value = list(self.value)
            else:
                value = self.value

            if value is not None and not (len(value) in (1, img.shape[-3])):
                raise ValueError(
                    "If value is a sequence, it should have either a single value or "
                    f"{img.shape[-3]} (number of input channels)"
                )

            x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=value)
            return F.erase(img, x, y, h, w, v, self.inplace)
        return img

    def randomize_parameters(self):
        pass


class TenCrop(object):
    """Crop the given image into four corners and the central crop plus the flipped version of
    these (horizontal flipping is used by default).
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
        vertical_flip (bool): Use vertical flipping instead of horizontal

    Example:
         >>> transform = Compose([
         >>>    TenCrop(size), # this is a list of PIL Images
         >>>    Lambda(lambda crops: torch.stack([PILToTensor()(crop) for crop in crops])) # returns a 4D tensor
         >>> ])
         >>> #In your test loop you can do the following:
         >>> input, target = batch # input is a 5d tensor, target is 2d
         >>> bs, ncrops, c, h, w = input.size()
         >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
         >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
    """

    def __init__(self, size, vertical_flip=False):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")
        self.vertical_flip = vertical_flip

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            tuple of 10 images. Image can be PIL Image or Tensor
        """
        return F.ten_crop(img, self.size, self.vertical_flip)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, vertical_flip={self.vertical_flip})"

    def randomize_parameters(self):
        pass


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size
