import torch
import numpy as np
import math

from Common.image_tools.resizer import resize_image, ResizingType

from core import models


AWING_MODEL_PATH = "/home/ignas/ml/Software/AdaptiveWingLoss/ckpt/WFLW_4HG.pth"
MODEL_NAME = "AWing"
HG_BLOCKS = 4
END_RELU = False
GRAY_SCALE = False
NUM_LANDMARKS = 98
INPUT_SIZE = 256


def get_AWing_landmark_model(device='cuda:0'):
    model = models.FAN(HG_BLOCKS, END_RELU, GRAY_SCALE, NUM_LANDMARKS)
    checkpoint = torch.load(AWING_MODEL_PATH)

    pretrained_weights = checkpoint['state_dict']
    model_weights = model.state_dict()
    pretrained_weights = {k: v for k, v in pretrained_weights.items() if k in model_weights}

    model_weights.update(pretrained_weights)
    model.load_state_dict(model_weights)
    model = model.to(device)

    def prediction_fn(inputs):
        outputs, boundary_channels = model(torch.Tensor(inputs).to(device))
        last_output = outputs[-1]
        pred_heatmaps = last_output[:, :NUM_LANDMARKS, :, :].detach().cpu()

        pred_landmarks, pred_landmark_confs = decode_landmark_heatmaps(pred_heatmaps)

        return pred_landmarks, pred_landmark_confs

    return model, MODEL_NAME, prediction_fn, NUM_LANDMARKS


def decode_landmark_heatmaps(hm):
    max, idx = torch.max(hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

    confidence = np.zeros([hm.size(0), hm.size(1), 1])

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]

            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            confidence[i, j] = hm_.detach().numpy()[pY, pX]
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = torch.FloatTensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    preds.add_(-0.5)
    preds = preds.numpy()
    confidence = np.squeeze(confidence, axis=2)

    return preds, confidence


def preprocess_image_for_landmarks(image, force_dims=True, transpose=True):
    image = resize_image(image, desired_shape=(INPUT_SIZE, INPUT_SIZE), resizing_type=ResizingType.FIXED)

    preprocessed_image = image / 255.

    if force_dims:
        input_batch = np.expand_dims(preprocessed_image, axis=0) if len(preprocessed_image.shape) < 4 else preprocessed_image
    else:
        input_batch = preprocessed_image

    if transpose:
        input_batch = np.transpose(input_batch, [0, 3, 1, 2]) if len(input_batch.shape) == 4 else np.transpose(input_batch, [2, 0, 1])

    return input_batch