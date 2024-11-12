# Copyright (C) 2023 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import json
import base64
from PIL import Image
import io
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

class ModelHandler:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sam_checkpoint = "/opt/nuclio/sam/sam_vit_h_4b8939.pth"
        self.model_type = "vit_h"
        self.latest_image = None
        sam_model = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam_model.to(device=self.device)
        self.predictor = SamPredictor(sam_model)

    def handle(self, image):
        self.predictor.set_image(np.array(image))
        features = self.predictor.get_image_embedding()
        return features


def init_context(context):
    context.logger.info("Init context...  0%")
    model = ModelHandler()
    context.user_data.model = model
    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("call handler")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf)
    image = image.convert("RGB")  #  to make sure image comes in RGB
    features = context.user_data.model.handle(image)

    return context.Response(body=json.dumps({
            'blob': base64.b64encode((features.cpu().numpy() if features.is_cuda else features.numpy())).decode(),
        }),
        headers={},
        content_type='application/json',
        status_code=200
    )
