import random

import numpy as np
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image):
        image = image.convert("L")
        try:
            import cv2

            arr = np.asarray(image)
            out = cv2.createCLAHE(self.clip_limit, self.tile_grid_size).apply(arr)
            return Image.fromarray(out).convert("RGB")
        except ImportError:
            return Image.fromarray(self._fallback(np.asarray(image))).convert("RGB")

    def _fallback(self, arr):
        out = np.empty_like(arr)
        h, w = arr.shape
        xs = np.linspace(0, w, self.tile_grid_size[0] + 1, dtype=int)
        ys = np.linspace(0, h, self.tile_grid_size[1] + 1, dtype=int)
        for y0, y1 in zip(ys[:-1], ys[1:]):
            for x0, x1 in zip(xs[:-1], xs[1:]):
                tile = arr[y0:y1, x0:x1]
                hist = np.bincount(tile.ravel(), minlength=256).astype(np.float32)
                limit = max(1, int(self.clip_limit * tile.size / 256))
                excess = np.maximum(hist - limit, 0).sum()
                hist = np.minimum(hist, limit) + excess / 256
                cdf = np.cumsum(hist)
                out[y0:y1, x0:x1] = np.clip(255 * cdf[tile] / cdf[-1], 0, 255)
        return out.astype(np.uint8)


class VeinIntensityJitter:
    def __init__(self, brightness=(0.75, 1.25), contrast=(0.8, 1.2), gamma=(0.8, 1.25), p=0.8):
        self.brightness = brightness
        self.contrast = contrast
        self.gamma = gamma
        self.p = p

    def __call__(self, image):
        if random.random() > self.p:
            return image
        arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        arr = np.clip(arr * random.uniform(*self.brightness), 0.0, 1.0)
        mean = arr.mean()
        arr = np.clip((arr - mean) * random.uniform(*self.contrast) + mean, 0.0, 1.0)
        arr = np.power(arr, random.uniform(*self.gamma))
        return Image.fromarray((arr * 255).astype(np.uint8)).convert("RGB")


def build_palm_transform(img_size, train=False):
    ops = [transforms.Grayscale(3), transforms.Resize((img_size, img_size))]
    if train:
        ops += [
            transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.98, 1.02)),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
        ]
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def build_vein_transform(img_size, train=False):
    ops = [transforms.Grayscale(3), transforms.Resize((img_size, img_size)), CLAHE()]
    if train:
        ops += [
            VeinIntensityJitter(),
            transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05)),
        ]
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
