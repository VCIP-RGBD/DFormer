from .. import *

# Dataset config
"""Dataset Path"""
C.dataset_name = "SUNRGBD"
C.dataset_path = osp.join(C.root_dir, "SUNRGBD")
C.rgb_root_folder = osp.join(C.dataset_path, "RGB")
C.rgb_format = ".jpg"
C.gt_root_folder = osp.join(C.dataset_path, "labels")
C.gt_format = ".png"
C.gt_transform = True
# True when label 0 is invalid, you can also modify the function _transform_gt in dataloader.RGBXDataset
# True for most dataset valid, Faslse for MFNet(?)
C.x_root_folder = osp.join(C.dataset_path, "Depth")
C.x_format = ".png"
C.x_is_single_channel = True  # True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = True
C.num_train_imgs = 5285
C.num_eval_imgs = 5050
C.num_classes = 37
C.class_names = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "blinds",
    "desk",
    "shelves",
    "curtain",
    "dresser",
    "pillow",
    "mirror",
    "floor_mat",
    "clothes",
    "ceiling",
    "books",
    "fridge",
    "tv",
    "paper",
    "towel",
    "shower_curtain",
    "box",
    "whiteboard",
    "person",
    "night_stand",
    "toilet",
    "sink",
    "lamp",
    "bathtub",
    "bag",
]

"""Image Config"""
C.background = 255
C.image_height = 480
C.image_width = 480
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])
