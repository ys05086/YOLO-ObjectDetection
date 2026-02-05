import os

import cv2
import numpy as np
import torch
import xml.etree.ElementTree as ET

from tqdm import tqdm
import sys
import torch.nn.functional as F

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
        test_image = np.zeros((1, size, size, 3), dtype = np.float32)
        test_img, test_gt = self.return_img_gt(
            index = index,
            grid_size = grid_size,
            size = size,
            bbox_count = bbox_count,
            classes = classes,
            test = True
        ) # test image shape = [S, S, 3]
        test_image[0:1, :, :, :] = test_img
        test_image = np.transpose(test_image, (0, 3, 1, 2)) / 255
        return test_image, test_gt

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

    def to_abs_coord(self, pred, gt):
        B, S = pred.shape[0], pred.shape[1]

        g_e = torch.arange(S, dtype = pred.dtype).to(pred.device)
        grid_x, grid_y = torch.meshgrid(g_e, g_e, indexing = 'ij')

        grid_x = grid_x.view(1, S, S, 1)
        grid_y = grid_y.view(1, S, S, 1)

        pred = pred[:, :, :, : self.bbox * 5]
        pred_boxes = torch.reshape(pred, (B, S, S, self.bbox, 5)) # [B, S, S, 2, 5]
        pred_bbox = pred_boxes[..., :4] # [B, S, S, 2, 4]
        gt_bbox = gt[..., :4].unsqueeze(-2) # [B, S, S, 1, (x, y, w, h)]

        # to abs coordinate
        ix = (pred_bbox[..., 0] + grid_x) / S
        iy = (pred_bbox[..., 1] + grid_y) / S
        iw = pred_bbox[..., 2]
        ih = pred_bbox[..., 3]

        gx = (gt_bbox[..., 0] + grid_x) / S
        gy = (gt_bbox[..., 1] + grid_y) / S
        gw = gt_bbox[..., 2]
        gh = gt_bbox[..., 3]

        ix1, iy1 = ix - iw / 2, iy - ih / 2
        ix2, iy2 = ix + iw / 2, iy + ih / 2

        gx1, gy1 = gx - gw / 2, gy - gh / 2
        gx2, gy2 = gx + gw / 2, gy + gh / 2

        list_ = [ix, iy, iw, ih, gx, gy, gw, gh]
        list_decode = [ix1, iy1, ix2, iy2, gx1, gx2, gy1, gy2]

        return list_, list_decode

    def tensor_iou(self, list_decode, gt, eps = 1e-7):
        obj_mask = gt[..., 4].unsqueeze(-1)  # [B, S, S, 1]

        ix1, iy1, ix2, iy2, gx1, gx2, gy1, gy2 = list_decode

        # intersection
        inx1 = torch.maximum(ix1, gx1)
        iny1 = torch.maximum(iy1, gy1)
        inx2 = torch.minimum(ix2, gx2)
        iny2 = torch.minimum(iy2, gy2)

        inw = torch.clamp(inx2 - inx1, min = 0.0)
        inh = torch.clamp(iny2 - iny1, min = 0.0)

        inter = inw * inh

        # union
        area_p = torch.clamp(ix2 - ix1, min = 0.0) * torch.clamp(iy2 - iy1, min = 0.0)
        area_g = torch.clamp(gx2 - gx1, min = 0.0) * torch.clamp(gy2 - gy1, min = 0.0)
        union = area_p + area_g - inter

        iou = inter / (union + eps) # to prevent division by 0, add eps to denom
        # we only need object in
        iou = iou * obj_mask

        return iou

    def forward(self, pred, gt, eps = 1e-7):
        list_, list_decode = self.to_abs_coord(pred, gt)

        ix, iy, iw, ih, gx, gy, gw, gh = list_ # [B, S, S, 2], [B, S, S]
        iou_tensor = self.tensor_iou(list_decode, gt, eps) # [B, S, S, 2]

        # make tensor size equal to: [B, S, S] -> [B, S, S, 2]
        gx = gx.expand_as(ix)
        gy = gy.expand_as(iy)
        gw = gw.expand_as(iw)
        gh = gh.expand_as(ih)

        obj_cell = gt[..., 4].unsqueeze(-1).to(pred.dtype)
        one_obj = iou_tensor.argmax(dim = -1)
        resp = F.one_hot(one_obj, num_classes = self.bbox).to(pred.dtype).to(pred.device)

        one_obj = obj_cell * resp
        one_noobj = (1 - obj_cell).expand_as(one_obj)

        # class prediction
        pred_cls = pred[..., self.bbox * 5: self.bbox * 5 + self.classes] # [B, S, S, C]
        gt_cls = gt[..., self.bbox * 5: self.bbox * 5 + self.classes]


        loss_1 = self.lc * (
                one_obj * (
                (gx - ix)**2 + (gy - iy)**2 +
                (torch.sqrt(gw) - torch.sqrt(torch.clamp(iw, min = 0.0)))**2 +
                (torch.sqrt(gh) - torch.sqrt(torch.clamp(ih, min=0.0)))**2)
        ).sum()

        loss_2 = ((one_obj + self.ln * one_noobj) *
                  ((torch.cat((pred[..., 4:5], pred[..., 9:10]), dim = -1) - iou_tensor.detach())**2)
                  ).sum()

        loss_3 = (obj_cell * ((pred_cls - gt_cls)**2)).sum()

        loss = loss_1 + loss_2 + loss_3
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
