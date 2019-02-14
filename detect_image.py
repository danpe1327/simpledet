import cv2
import os
import argparse
import importlib
import mxnet as mx
import numpy as np

from core.detection_module import DetModule
from utils.load_model import load_checkpoint

CATEGORIES = [
    "__background",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def parse_args():
    parser = argparse.ArgumentParser(description='Test Detection')
    # general
    parser.add_argument('img', help='the image path', type=str)
    parser.add_argument('--config', help='config file path', type=str, default='config/tridentnet_r101v2c4_c5_1x.py')
    parser.add_argument('--batch_size', help='', type=int, default=1)
    parser.add_argument('--gpu', help='the gpu id for inferencing', type=int, default=0)
    parser.add_argument('--thresh', help='the threshold for filtering boxes', type=float, default=0.7)
    args = parser.parse_args()

    return args


class predictor(object):
    def __init__(self, config, batch_size, gpu_id, thresh):
        self.config = config
        self.batch_size = batch_size
        self.thresh = thresh

        # Parse the parameter file of model
        pGen, pKv, pRpn, pRoi, pBbox, pDataset, pModel, pOpt, pTest, \
        transform, data_name, label_name, metric_list = config.get_config(is_train=False)

        self.data_name = data_name
        self.label_name = label_name
        self.p_long, self.p_short = transform[1].p.long, transform[1].p.short

        # Define NMS type
        if callable(pTest.nms.type):
            self.do_nms = pTest.nms.type(pTest.nms.thr)
        else:
            from operator_py.nms import py_nms_wrapper

            self.do_nms = py_nms_wrapper(pTest.nms.thr)

        sym = pModel.test_symbol
        sym.save(pTest.model.prefix + "_test.json")

        ctx = mx.gpu(gpu_id)
        data_shape = [
            ('data', (batch_size, 3, 800, 1200)),
            ("im_info", (1, 3)),
            ("im_id", (1,)),
            ("rec_id", (1,)),
        ]

        # Load network
        arg_params, aux_params = load_checkpoint(pTest.model.prefix, pTest.model.epoch)
        self.mod = DetModule(sym, data_names=data_name, context=ctx)
        self.mod.bind(data_shapes=data_shape, for_training=False)
        self.mod.set_params(arg_params, aux_params, allow_extra=False)

    def preprocess_image(self, input_img):
        image = input_img[:, :, ::-1]  # BGR -> RGB

        short = min(image.shape[:2])
        long = max(image.shape[:2])
        scale = min(self.p_short / short, self.p_long / long)

        h, w = image.shape[:2]
        im_info = (round(h * scale), round(w * scale), scale)

        image = cv2.resize(image, None, None, scale, scale, interpolation=cv2.INTER_LINEAR)
        image = image.transpose((2, 0, 1))  # HWC -> CHW

        return image, im_info

    def run_image(self, img_path):
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image, im_info = self.preprocess_image(image)
        input_data = {'data': [image],
                      'im_info': [im_info],
                      'im_id': [0],
                      'rec_id': [0],
                      }

        data = [mx.nd.array(input_data[name]) for name in self.data_name]
        label = []
        provide_data = [(k, v.shape) for k, v in zip(self.data_name, data)]
        provide_label = [(k, v.shape) for k, v in zip(self.label_name, label)]

        data_batch = mx.io.DataBatch(data=data,
                                     label=label,
                                     provide_data=provide_data,
                                     provide_label=provide_label)

        self.mod.forward(data_batch, is_train=False)
        out = [x.asnumpy() for x in self.mod.get_outputs()]

        cls_score = out[3]
        bboxes = out[4]

        result = {}
        for cid in range(cls_score.shape[1]):
            if cid == 0:  # Ignore the background
                continue
            score = cls_score[:, cid]
            if bboxes.shape[1] != 4:
                cls_box = bboxes[:, cid * 4:(cid + 1) * 4]
            else:
                cls_box = bboxes
            valid_inds = np.where(score >= self.thresh)[0]
            box = cls_box[valid_inds]
            score = score[valid_inds]
            det = np.concatenate((box, score.reshape(-1, 1)), axis=1).astype(np.float32)
            det = self.do_nms(det)
            if len(det) > 0:
                det[:, :4] = det[:, :4] / im_info[2]  # Restore to the original size
                result[CATEGORIES[cid]] = det

        return result


if __name__ == "__main__":
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

    args = parse_args()
    img_path = args.img
    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    batch_size = args.batch_size
    gpu_id = args.gpu
    thresh = args.thresh
    
    save_dir = 'out'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    coco_predictor = predictor(config, batch_size, gpu_id, thresh)
    result = coco_predictor.run_image(img_path)

    draw_img = cv2.imread(img_path)
    for k, v in result.items():
        print('%s, num:%d' % (k, v.shape[0]))
        for box in v:
            score = box[4]
            box = box.astype(int)
            x1, y1, x2, y2 = box[:4]

            cv2.putText(draw_img, '%s:%.2f' % (k, score), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    save_name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(save_dir, 'result_%s' % save_name), draw_img)
