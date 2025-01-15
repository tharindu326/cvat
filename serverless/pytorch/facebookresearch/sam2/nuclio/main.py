# Copyright (C) 2023-2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import json
import base64
from PIL import Image
import io
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class ModelHandler:
    def __init__(self):
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            if torch.cuda.get_device_properties(0).major >= 8:
                # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        self.sam_checkpoint = "./sam2_hiera_large.pt"
        self.model_cfg = "sam2_hiera_l.yaml"
        self.predictor = SAM2ImagePredictor(build_sam2(self.model_cfg, self.sam_checkpoint, device=self.device))

    def handle(self, image, pos_points, neg_points):
        pos_points, neg_points = list(pos_points), list(neg_points)
        with torch.inference_mode():
            self.predictor.set_image(np.array(image))
            masks, scores, _ = self.predictor.predict(
                point_coords=np.array(pos_points + neg_points),
                point_labels=np.array([1]*len(pos_points) + [0]*len(neg_points)),
                multimask_output=True,
            )
            sorted_ind = np.argsort(scores)[::-1]
            best_mask = masks[sorted_ind][0]
            return best_mask

def init_context(context):
    model = ModelHandler()
    context.user_data.model = model
    context.logger.info("Init context...100%")

def handler(context, event):
    try:
        context.logger.info("call handler")
        data = event.body
        buf = io.BytesIO(base64.b64decode(data["image"]))
        image = Image.open(buf)
        image = image.convert("RGB")  # to make sure image comes in RGB
        pos_points = data["pos_points"]
        neg_points = data["neg_points"]

        mask = context.user_data.model.handle(image, pos_points, neg_points)

        return context.Response(
            body=json.dumps({'mask': mask.tolist()}),
            headers={},
            content_type='application/json',
            status_code=200
        )
    except Exception as e:
        context.logger.error(f"Error in handler: {str(e)}")
        return context.Response(
            body=json.dumps({'error': str(e)}),
            headers={},
            content_type='application/json',
            status_code=500
        )
