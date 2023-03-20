import os
import os.path as osp
import sys

import torch
import torch.nn.functional as F

sys.path.append(osp.join(osp.dirname(__file__), "../../.."))
from InteractionAOG_Image.src.datasets import get_dataset
from torchvision.transforms.functional import to_pil_image

# ===========================================
#             C o n f i g s
# ===========================================
dataset_name, version_tag, slots = "simplemnist", "2", {i: 10 for i in [3, 4, 5, 8, 0]}  # to specify the number of samples for each class


save_root = f"../../saved-manual-segments/{dataset_name}-segments/version-{version_tag}"
os.makedirs(save_root, exist_ok=True)
dataset = get_dataset("/data2/limingjie/data", dataset_name, eval_mode=True)
train_loader, _ = dataset.get_dataloader(batch_size=1)


# ===========================================
#               M a i n
# ===========================================
for sample_id, (image, label) in enumerate(train_loader):
    assert image.shape[0] == 1

    if sum(list(slots.values())) == 0:
        break

    label_id = label.item()
    if label_id in slots and slots[label_id] > 0:
        slots[label_id] -= 1
    else:
        continue

    os.makedirs(osp.join(save_root, str(label_id)), exist_ok=True)
    torch.save(image, osp.join(save_root, str(label_id), f"sample_{sample_id:>05d}_image.pth"))
    torch.save(label, osp.join(save_root, str(label_id), f"sample_{sample_id:>05d}_label.pth"))

    # image = denormalize_image(image, mean=[0.5], std=[0.5]) # no denormalization
    image = F.interpolate(image, scale_factor=32, mode="nearest")
    image = image.squeeze(0)

    image = to_pil_image(image)
    image.save(osp.join(save_root, str(label_id), f"sample_{sample_id:>05d}_hq.png"))
