import torch
import torch.nn as nn
import torchvision.models as models
import openslide
import numpy as np
import matplotlib.pyplot as plt

model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, 3)

weights = # Specify your weights 
model.load_state_dict(torch.load(weights, map_location='cpu'), strict = False)

slide_path = # Specify the slide path

coord = # Where to start the inference from
size = # Size of the zone to infer on

patch_size = # Size of the patch to chose
def wsi_inference(slide, coords, size):
    heatmap_mat = np.zeros((round(size[0] / patch_size) + 1, round(size[1] / patch_size) + 1))
    i = 0
    for x in range(coord[0], coord[0] + size[0], patch_size):
        j = 0
        for y in range(coord[1], coord[1] + size[1], patch_size):
            temp_img = np.asarray(slide.read_region((x, y), 0, (patch_size, patch_size)))[..., :3]
            temp_img = (temp_img - np.min(temp_img)) / (np.max(temp_img) - np.min(temp_img))

            tensor = torch.from_numpy(temp_img)

            tensor = tensor.float()

            # normalize tensor
            tensor = (tensor - tensor.mean()) / tensor.std()

            tensor = torch.permute(tensor, (2, 0, 1))

            tensor = tensor[None, :]
            output = model(tensor)
            _,pred = torch.max(output, dim=1)
            heatmap_mat[i][j] = pred.item()
            j += 1
        i += 1
    return heatmap_mat
  
# Test & Visualize
heatmap_map = wsi_inference(slide, coord, size)
plt.imshow(heatmap_mat, cmap='hot', interpolation='nearest')
plt.show()
