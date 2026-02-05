import os

import cv2
import numpy as np
import torch
import xml.etree.ElementTree as ET
from tqdm import tqdm
import sys

class_colormap = {
    "0":[0,0,0], #back ground
    "1":[0,0,128], #aero plane
    "2":[0,128,0], #Bicycle
    "3":[0,128,128], # Bird
    "4":[128,0,0], #Boat
    "5":[128,0,128], #Bottle
    "6":[128,128,0], #Bus
    "7":[128,128,128], #Car
    "8":[0,0,64], #Cat
    "9":[0,0,192], #Chair
    "10":[0,128,64], #Cow
    "11":[0,128,192], #Dining Table
    "12":[128,0,64], #Dog
    "13":[128,0,192], #Horse
    "14":[128,128,64], #Motorbike
    "15":[128,128,192], #Person
    "16":[0,64,0], #Potted Plant
    "17":[0,64,128], #Sheep
    "18":[0,192,0], #Sofa
    "19":[0,192,128], #Train
    "20":[128,64,0], #TV/Monitor
}

# def data_augmentation

class LoadData: # and Data preprocessing
    def __init__(self, annotation_path, image_path, size = 448, numbers = 1400, seed = 42, set_test_data = True):
        super(LoadData, self).__init__()
        self.annotation_path = annotation_path
        self.image_path = image_path

        # image data is also included here
        self.list_annotations = os.listdir(self.annotation_path)

        # change anno data
        self.anno_data = None
        self.set_data(size) # saved all the data by type: [name, size, x, y, w, h, class]

        # set test data
        self.test_data = None
        if set_test_data:
            self.set_test_data(seed, numbers)

    def data_augmentation(self, img, gt):
        return img, gt

    # def minibatch_training(self, batch_size):
    #     np.random.seed(None)
    #     img = read
    #     return img, gt_tensor

    # [ name, size, x_mid, y_mid, x_len, y_len, class ]
    # list_ = anno_data[index]
    def make_gt(self, grid_size: int, list_: list, bbox_count: int = 2, classes: int = 20):
        gt = np.zeros((grid_size, grid_size, 5 + classes), dtype = np.float32)
        # x, y , w, h, conf, class one hot
        # i am using

        if isinstance(list_[0], (list, tuple)):
            size = list_[0][1]
            for l in range(len(list_)):
                grid_i = min(grid_size - 1, max(0, int(list_[l][2] * grid_size // size)))
                grid_j = min(grid_size - 1, max(0, int(list_[l][3] * grid_size // size)))
                # grid_i = int(list_[l][2] * grid_size // size)
                # grid_j = int(list_[l][3] * grid_size // size)
                if gt[grid_j, grid_i, 4] == 0:
                    gt[grid_j, grid_i, 0] = (list_[l][2] * grid_size / size) - grid_i
                    gt[grid_j, grid_i, 1] = (list_[l][3] * grid_size / size) - grid_j
                    gt[grid_j, grid_i, 2] = list_[l][4] / size
                    gt[grid_j, grid_i, 3] = list_[l][5] / size
                    gt[grid_j, grid_i, 4] = 1  # confidence
                    gt[:, :, 5:] += self.index_to_onehot(list_[l][6], grid_size, grid_i, grid_j, classes)
                else:
                    continue
        else:
            size = list_[1]
            grid_i = min(grid_size - 1, max(0, int(list_[2] * grid_size // size)))
            grid_j = min(grid_size - 1, max(0, int(list_[3] * grid_size // size)))
            # grid_i = int(list_[2] * grid_size // size)
            # grid_j = int(list_[3] * grid_size // size)
            gt[grid_j, grid_i, 0] = (list_[2] * grid_size / size) - grid_i
            gt[grid_j, grid_i, 1] = (list_[3] * grid_size / size) - grid_j
            gt[grid_j, grid_i, 2] = list_[4] / size
            gt[grid_j, grid_i, 3] = list_[5] / size
            gt[grid_j, grid_i, 4] = 1 # confidence
            gt[:, :, 5 :] = self.index_to_onehot(list_[6], grid_size, grid_i, grid_j, classes)

        return gt

    def test_(self, index: int):
        print("Test Done.")
        img, gt_tensor = self.return_img_gt(index, grid_size = 7, size = 448, bbox_count = 2, classes = 20)
        print(gt_tensor)
        print(np.shape(gt_tensor))
        return img

    def index_to_onehot(self, index: int, grid_size: int, grid_i: int, grid_j: int, classes: int = 20):
        onehot = np.zeros((grid_size, grid_size, classes), dtype = np.float32)
        onehot[grid_j, grid_i, index] = 1
        return onehot

    def to_abs(self, value, size, grid_size):
        return value * grid_size // size

    def return_img_gt(self, index: int, grid_size: int, size: int, bbox_count: int = 2, classes: int = 20, test: bool = False):
        if not test:
            list_ = self.anno_data[index]
        else:
            list_ = self.test_data[index]
        gt_tensor = self.make_gt(grid_size, list_, bbox_count, classes)
        img = self.read_image(index, size)
        return img, gt_tensor

    def read_xml(self, index: int, size: int):
        i = index
        self.check_path()
        # load xml
        tree = ET.parse(os.path.join(self.annotation_path, self.list_annotations[i]))
        root = tree.getroot()

        # list for temporary save
        list_ = []
        W = int(root.find('size').findtext('width'))
        H = int(root.find('size').findtext('height'))

        lambda_w = size / W
        lambda_h = size / H

        name = root.find('filename').text #.text[:-5]

        for object in root.iter('object'):
            class_ = object.find('name').text
            x_min = int(object.find('bndbox').findtext('xmin'))
            y_min = int(object.find('bndbox').findtext('ymin'))
            x_max = int(object.find('bndbox').findtext('xmax'))
            y_max = int(object.find('bndbox').findtext('ymax'))
            x_mid = (x_max + x_min) * lambda_w / 2
            y_mid = (y_max + y_min) * lambda_h / 2
            x_len = (x_max - x_min) * lambda_w
            y_len = (y_max - y_min) * lambda_h
            # list_.append([int(x_mid), int(y_mid), int(x_len), int(y_len), self.class_to_index(class_)])
            list_.append([name, size, x_mid, y_mid, x_len, y_len, self.class_to_index(class_)])
        return list_

    def load_test_data(self, index: int, grid_size: int, size: int = 448, bbox_count: int = 2, classes: int = 20):
        i = index
        test_img, test_gt = self.return_img_gt(
            i,
            grid_size = grid_size,
            size = size,
            bbox_count = bbox_count,
            classes = classes
        )
        return test_img, test_gt

    def set_data(self, size: int = 448):
        data = []
        for i in tqdm(range(len(self.list_annotations)), ncols = 120, desc = "Loading Raw Data.."):
            data.append(self.read_xml(i, size))
        self.anno_data = data
        print("Loading Done.")

    def set_test_data(self, seed: int, numbers: int):
        np.random.seed(seed)
        self.test_data = []
        rand_num = np.random.choice(len(self.list_annotations), numbers, replace = False)
        index = sorted(rand_num, reverse = True)
        for i in range(numbers):
            self.test_data.append(self.anno_data.pop(index[i]))
        print("Test Data Seperated")

    def minibatch_training(self, batch_size: int, grid_size: int, size: int = 448, bbox_count: int = 2, classes: int = 20):
        rand_num = np.random.randint(0, len(self.anno_data), batch_size)
        batch_imgs = np.zeros((batch_size, size, size, 3))
        batch_gts = np.zeros((batch_size, grid_size, grid_size, 5 + classes))
        for it in range(len(rand_num)):
            temp_img, temp_gt = self.return_img_gt(rand_num[it], grid_size = grid_size, size = size, bbox_count = 2, classes = 20)
            batch_imgs[it, : ,:, :] = temp_img
            batch_gts[it, : ,:, :] = temp_gt

        batch_imgs = np.transpose(batch_imgs, (0, 3, 1, 2)) / 255
        # batch_gts = np.transpose(batch_gts, (0, 3, 1, 2))
        return batch_imgs, batch_gts

    def read_image(self, index: int, size: int, test: bool = False):
        if not test:
            anno = self.anno_data[index]
        else:
            anno = self.test_data[index]
        self.check_path()

        filename = anno[0][0] if isinstance(anno[0], (list, tuple)) else anno[0]
        # noinspection PyTypeChecker
        image = cv2.imread(os.path.join(self.image_path, filename))
        if image is None:
            print("!!IMAGE NOT FOUND!!")
            sys.exit()
        image = cv2.resize(
            image,
            (size, size),
            interpolation = cv2.INTER_LINEAR
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR to RGB
        return image

    # def read_batch_image(self, batch_size, size, seed = None):
    #     np.random.seed(seed)
    #     batch_images = np.zeros((batch_size, size, size, 3))
    #     rand_num = np.random.randint(0, len(self.list_annotations), batch_size)
    #     for it in range(len(rand_num)):
    #         temp_img = self.read_image(it, size, self.anno_data[rand_num[it]][0])
    #         batch_images[it, :, :, :] = temp_img
    #     return batch_images, rand_num

    def check_path(self):
        # check end of path whether it contains '/' or not
        if self.annotation_path[-1] != '/':
            self.annotation_path += '/'
        if self.image_path[-1] != '/':
            self.image_path += '/'

    def class_to_index(self, class_):
        index = {
            "aeroplane": 0,
            "bicycle": 1,
            "bird": 2,
            "boat": 3,
            "bottle": 4,
            "bus": 5,
            "car": 6,
            "cat": 7,
            "chair": 8,
            "cow": 9,
            "diningtable": 10,
            "dog": 11,
            "horse": 12,
            "motorbike": 13,
            "person": 14,
            "pottedplant": 15,
            "sheep": 16,
            "sofa": 17,
            "train": 18,
            "tvmonitor": 19
        }
        return index[class_]


# noinspection SpellCheckingInspection
class LossFunction(torch.nn.Module):
    def __init__(self, lambda_coord = 5, lambda_noobj = 0.5, size = 448, bbox_count = 2, classes = 20):
        super(LossFunction, self).__init__()
        self.lc = lambda_coord
        self.ln = lambda_noobj
        self.bbox = bbox_count
        self.classes = classes
        self.size = size

    def check(self, i):
        one_obj = i
        return one_obj

    def confidence(self, tensor):
        confidence = 0.1
        return confidence

    def to_global_coordinate(self, tensor, size = 448):
        return tensor

    def mask(self, gt):
        obj_mask = gt[:, :, :, 4]
        noobj_mask = torch.where(obj_mask == 0, self.ln, 0)
        return obj_mask + noobj_mask

    def reverse_(self, value):
        return 0

    def tensor_iou(self, pred, gt):
        B, S = pred.shape[0], pred.shape[1]

        g_e = torch.tensor([i for i in range(S)]) # grid element
        grid_x, grid_y = torch.meshgrid(g_e, g_e, indexing='ij')

        pred = pred[:, :, :, : self.bbox * 5]
        pred_boxes = torch.reshape(pred, (B, S, S, self.bbox, 5))
        pred_bbox = pred_boxes[..., :4]
        gt_bbox = gt[..., :4].unsqueeze(-2) # [B, S, S, 1, (x, y, w, h)]
        obj_mask = gt[..., 4].unsqueeze(-1) # [B, S, S, 1]
        

        return tensor

    def pre(self, pred, gt):

        return pred, gt, iou_tensor, one_tensor

    def forward(self, pred, gt):
        loss_1 = self.lc * ((gt[:, :, 0] - pred[:, :, 0])**2 + (gt[:, :, 1] - pred[:, :, 1])**2).sum(dim = 1)
        loss_2 = self.lc * (((torch.sqrt(gt[:, :, 2]) - torch.sqrt(pred[:, :, 2]))**2).sum(dim = 1) + ((torch.sqrt(gt[:, :, 3]) - torch.sqrt(pred[:, :, 3]))**2)).sum(dim = 1)
        loss_3 = ((self.confidence(gt) - self.confidence(pred))**2).sum()
        loss_4 = self.ln * ((self.confidence(gt) - self.confidence(pred))**2).sum()
        loss_5 = 1
        loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5
        return loss

# image visualisation and show bbox
# def visualize(img, pred, gt):

# for saving
class Tee:
    def __init__(self, console, file):
        self.file = file
        self.console = console

    def write(self, message):
        if '\r' in message:
            self.console.write(message)
        else:
            self.console.write(message)
            self.file.write(message)
            self.file.flush()

    def flush(self):
        self.console.flush()
        self.file.flush()
