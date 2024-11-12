import base64
import io
import json
import yaml
from PIL import Image
import cv2
import numpy as np
import onnxruntime as ort


class ModelHandler:
    def __init__(self, labels):
        self.model = None
        self.load_network(model="yolov7-nms-640.onnx")
        self.labels = labels

    def load_network(self, model):
        device = ort.get_device()
        cuda = True if device == 'GPU' else False
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            so = ort.SessionOptions()
            so.log_severity_level = 3

            self.model = ort.InferenceSession(model, providers=providers, sess_options=so)
            self.output_details = [i.name for i in self.model.get_outputs()]
            self.input_details = [i.name for i in self.model.get_inputs()]

            self.is_inititated = True
        except Exception as e:
            raise Exception(f"Cannot load model {model}: {e}")

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    def _infer(self, inputs: np.ndarray):
        try:
            img = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)
            image = img.copy()
            image, ratio, dwdh = self.letterbox(image, auto=False)
            image = image.transpose((2, 0, 1))
            image = np.expand_dims(image, 0)
            image = np.ascontiguousarray(image)

            im = image.astype(np.float32)
            im /= 255

            inp = {self.input_details[0]: im}
            # ONNX inference
            output = list()
            detections = self.model.run(self.output_details, inp)[0]

            # for det in detections:
            boxes = detections[:, 1:5]
            labels = detections[:, 5]
            scores = detections[:, -1]

            boxes -= np.array(dwdh * 2)
            boxes /= ratio
            boxes = boxes.round().astype(np.int32)
            output.append(boxes)
            output.append(labels)
            output.append(scores)
            return output

        except Exception as e:
            print(e)

    def infer(self, image, threshold):
        image = np.array(image)
        image = image[:, :, ::-1].copy()
        h, w, _ = image.shape
        detections = self._infer(image)

        results = []
        if detections:
            boxes = detections[0]
            labels = detections[1]
            scores = detections[2]

            for label, score, box in zip(labels, scores, boxes):
                if score >= threshold:
                    xtl = max(int(box[0]), 0)
                    ytl = max(int(box[1]), 0)
                    xbr = min(int(box[2]), w)
                    ybr = min(int(box[3]), h)

                    results.append({
                        "confidence": str(score),
                        "label": self.labels.get(label, "unknown"),
                        "points": [xtl, ytl, xbr, ybr],
                        "type": "rectangle",
                    })

        return results


def init_context(context):
    context.logger.info("Init context...  0%")

    # Read labels
    # with open("/opt/nuclio/function.yaml", 'rb') as function_file:
    #     functionconfig = yaml.safe_load(function_file)

    labels_spec = [
        { "id": 0, "name": "person", "type": "rectangle" },
        { "id": 1, "name": "bicycle", "type": "rectangle" },
        { "id": 2, "name": "car", "type": "rectangle" },
        { "id": 3, "name": "motorbike", "type": "rectangle" },
        { "id": 4, "name": "aeroplane", "type": "rectangle" },
        { "id": 5, "name": "bus", "type": "rectangle" },
        { "id": 6, "name": "train", "type": "rectangle" },
        { "id": 7, "name": "truck", "type": "rectangle" },
        { "id": 8, "name": "boat", "type": "rectangle" },
        { "id": 9, "name": "traffic light", "type": "rectangle" },
        { "id": 10, "name": "fire hydrant", "type": "rectangle" },
        { "id": 11, "name": "stop sign", "type": "rectangle" },
        { "id": 12, "name": "parking meter", "type": "rectangle" },
        { "id": 13, "name": "bench", "type": "rectangle" },
        { "id": 14, "name": "bird", "type": "rectangle" },
        { "id": 15, "name": "cat", "type": "rectangle" },
        { "id": 16, "name": "dog", "type": "rectangle" },
        { "id": 17, "name": "horse", "type": "rectangle" },
        { "id": 18, "name": "sheep", "type": "rectangle" },
        { "id": 19, "name": "cow", "type": "rectangle" },
        { "id": 20, "name": "elephant", "type": "rectangle" },
        { "id": 21, "name": "bear", "type": "rectangle" },
        { "id": 22, "name": "zebra", "type": "rectangle" },
        { "id": 23, "name": "giraffe", "type": "rectangle" },
        { "id": 24, "name": "backpack", "type": "rectangle" },
        { "id": 25, "name": "umbrella", "type": "rectangle" },
        { "id": 26, "name": "handbag", "type": "rectangle" },
        { "id": 27, "name": "tie", "type": "rectangle" },
        { "id": 28, "name": "suitcase", "type": "rectangle" },
        { "id": 29, "name": "frisbee", "type": "rectangle" },
        { "id": 30, "name": "skis", "type": "rectangle" },
        { "id": 31, "name": "snowboard", "type": "rectangle" },
        { "id": 32, "name": "sports ball", "type": "rectangle" },
        { "id": 33, "name": "kite", "type": "rectangle" },
        { "id": 34, "name": "baseball bat", "type": "rectangle" },
        { "id": 35, "name": "baseball glove", "type": "rectangle" },
        { "id": 36, "name": "skateboard", "type": "rectangle" },
        { "id": 37, "name": "surfboard", "type": "rectangle" },
        { "id": 38, "name": "tennis racket", "type": "rectangle" },
        { "id": 39, "name": "bottle", "type": "rectangle" },
        { "id": 40, "name": "wine glass", "type": "rectangle" },
        { "id": 41, "name": "cup", "type": "rectangle" },
        { "id": 42, "name": "fork", "type": "rectangle" },
        { "id": 43, "name": "knife", "type": "rectangle" },
        { "id": 44, "name": "spoon", "type": "rectangle" },
        { "id": 45, "name": "bowl", "type": "rectangle" },
        { "id": 46, "name": "banana", "type": "rectangle" },
        { "id": 47, "name": "apple", "type": "rectangle" },
        { "id": 48, "name": "sandwich", "type": "rectangle" },
        { "id": 49, "name": "orange", "type": "rectangle" },
        { "id": 50, "name": "broccoli", "type": "rectangle" },
        { "id": 51, "name": "carrot", "type": "rectangle" },
        { "id": 52, "name": "hot dog", "type": "rectangle" },
        { "id": 53, "name": "pizza", "type": "rectangle" },
        { "id": 54, "name": "donut", "type": "rectangle" },
        { "id": 55, "name": "cake", "type": "rectangle" },
        { "id": 56, "name": "chair", "type": "rectangle" },
        { "id": 57, "name": "sofa", "type": "rectangle" },
        { "id": 58, "name": "pottedplant", "type": "rectangle" },
        { "id": 59, "name": "bed", "type": "rectangle" },
        { "id": 60, "name": "diningtable", "type": "rectangle" },
        { "id": 61, "name": "toilet", "type": "rectangle" },
        { "id": 62, "name": "tvmonitor", "type": "rectangle" },
        { "id": 63, "name": "laptop", "type": "rectangle" },
        { "id": 64, "name": "mouse", "type": "rectangle" },
        { "id": 65, "name": "remote", "type": "rectangle" },
        { "id": 66, "name": "keyboard", "type": "rectangle" },
        { "id": 67, "name": "cell phone", "type": "rectangle" },
        { "id": 68, "name": "microwave", "type": "rectangle" },
        { "id": 69, "name": "oven", "type": "rectangle" },
        { "id": 70, "name": "toaster", "type": "rectangle" },
        { "id": 71, "name": "sink", "type": "rectangle" },
        { "id": 72, "name": "refrigerator", "type": "rectangle" },
        { "id": 73, "name": "book", "type": "rectangle" },
        { "id": 74, "name": "clock", "type": "rectangle" },
        { "id": 75, "name": "vase", "type": "rectangle" },
        { "id": 76, "name": "scissors", "type": "rectangle" },
        { "id": 77, "name": "teddy bear", "type": "rectangle" },
        { "id": 78, "name": "hair drier", "type": "rectangle" },
        { "id": 79, "name": "toothbrush", "type": "rectangle" }
      ]
    # labels = {item['id']: item['name'] for item in json.loads(labels_spec)}
    labels = {item['id']: item['name'] for item in labels_spec}

    # Read the DL model
    model = ModelHandler(labels)
    context.user_data.model = model

    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Run YoloV7 ONNX model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    image = Image.open(buf)

    results = context.user_data.model.infer(image, threshold)

    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)
