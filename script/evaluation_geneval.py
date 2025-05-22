import os
import json
import numpy as np
from PIL import Image, ImageOps
import torch
import warnings
warnings.filterwarnings("ignore")

import mmdet
from mmdet.apis import inference_detector, init_detector
import open_clip
from clip_benchmark.metrics import zeroshot_classification as zsc
zsc.tqdm = lambda it, *args, **kwargs: it  # remove tqdm from classification

# ==== Default Settings ====
THRESHOLD = 0.3
COUNTING_THRESHOLD = 0.9
MAX_OBJECTS = 16
NMS_THRESHOLD = 1.0
POSITION_THRESHOLD = 0.1
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"

# ==== Load Once ====
print("[Geneval Eval] Loading detection and CLIP models...")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"))
CKPT_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../geneval/<OBJECT_DETECTOR_FOLDER>/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"))

object_detector = init_detector(CONFIG_PATH, CKPT_PATH, device=DEVICE)

CLIP_ARCH = "ViT-L-14"
clip_model, _, transform = open_clip.create_model_and_transforms(CLIP_ARCH, pretrained="openai", device=DEVICE)
tokenizer = open_clip.get_tokenizer(CLIP_ARCH)

with open("geneval/evaluation/object_names.txt") as f:
    classnames = [line.strip() for line in f]

print("[Geneval Eval] Models loaded successfully.")


# ==== Evaluation Function ====
def evaluate_geneval_image(image_path, metadata):
    def compute_iou(box_a, box_b):
        area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
        i_area = area_fn([
            max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
            min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
        ])
        u_area = area_fn(box_a) + area_fn(box_b) - i_area
        return i_area / u_area if u_area else 0

    def relative_position(obj_a, obj_b):
        boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
        center_a, center_b = boxes.mean(axis=-2)
        dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
        offset = center_a - center_b
        revised_offset = np.maximum(np.abs(offset) - POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
        if np.all(np.abs(revised_offset) < 1e-3):
            return set()
        dx, dy = revised_offset / np.linalg.norm(offset)
        relations = set()
        if dx < -0.5: relations.add("left of")
        if dx > 0.5: relations.add("right of")
        if dy < -0.5: relations.add("above")
        if dy > 0.5: relations.add("below")
        return relations

    def color_classification(image, bboxes, classname):
        COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
        classifier = zsc.zero_shot_classifier(
            clip_model, tokenizer, COLORS,
            [
                f"a photo of a {{c}} {classname}",
                f"a photo of a {{c}}-colored {classname}",
                f"a photo of a {{c}} object"
            ],
            next(clip_model.parameters()).device
        )
        class ImageCrops(torch.utils.data.Dataset):
            def __init__(self, image: Image.Image, objects):
                self._image = image.convert("RGB")
                self._blank = Image.new("RGB", image.size, color="#999")
                self._objects = objects
            def __len__(self): return len(self._objects)
            def __getitem__(self, index):
                box, mask = self._objects[index]
                crop_img = self._image.crop(box[:4])
                return (transform(crop_img), 0)

        loader = torch.utils.data.DataLoader(ImageCrops(image, bboxes), batch_size=16, num_workers=2)
        with torch.no_grad():
            pred, _ = zsc.run_classification(clip_model, classifier, loader, next(clip_model.parameters()).device)
            return [COLORS[i.item()] for i in pred.argmax(1)]

    # Run detector
    result = inference_detector(object_detector, image_path)
    bbox = result[0] if isinstance(result, tuple) else result
    segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
    image = ImageOps.exif_transpose(Image.open(image_path))
    detected = {}

    conf_thresh = COUNTING_THRESHOLD if metadata.get("tag") == "counting" else THRESHOLD
    for index, classname in enumerate(classnames):
        ordering = np.argsort(bbox[index][:, 4])[::-1]
        ordering = ordering[bbox[index][ordering, 4] > conf_thresh]
        ordering = ordering[:MAX_OBJECTS].tolist()
        if not ordering:
            continue
        detected[classname] = []
        while ordering:
            max_obj = ordering.pop(0)
            detected[classname].append((bbox[index][max_obj], None if segm is None else segm[index][max_obj]))
            ordering = [
                obj for obj in ordering
                if NMS_THRESHOLD == 1 or compute_iou(bbox[index][max_obj], bbox[index][obj]) < NMS_THRESHOLD
            ]

    # Evaluate
    correct = True
    reason = []
    matched_groups = []

    for req in metadata.get('include', []):
        classname = req['class']
        found_objects = detected.get(classname, [])[:req['count']]
        matched = True
        if len(found_objects) < req['count']:
            correct = matched = False
            reason.append(f"expected {classname}>={req['count']}, found {len(found_objects)}")
        else:
            if 'color' in req:
                colors = color_classification(image, found_objects, classname)
                if colors.count(req['color']) < req['count']:
                    correct = matched = False
                    reason.append(
                        f"expected {req['color']} {classname}>={req['count']}, found " +
                        f"{colors.count(req['color'])} {req['color']}; and " +
                        ", ".join(f"{colors.count(c)} {c}" for c in set(colors))
                    )
            if 'position' in req and matched:
                expected_rel, target_group = req['position']
                if matched_groups[target_group] is None:
                    correct = matched = False
                    reason.append(f"no target for {classname} to be {expected_rel}")
                else:
                    for obj in found_objects:
                        for target_obj in matched_groups[target_group]:
                            true_rels = relative_position(obj, target_obj)
                            if expected_rel not in true_rels:
                                correct = matched = False
                                reason.append(
                                    f"expected {classname} {expected_rel} target, found " +
                                    f"{' and '.join(true_rels)} target"
                                )
                                break
                        if not matched:
                            break
        matched_groups.append(found_objects if matched else None)

    for req in metadata.get('exclude', []):
        classname = req['class']
        if len(detected.get(classname, [])) >= req['count']:
            correct = False
            reason.append(f"expected {classname}<{req['count']}, found {len(detected[classname])}")

    return {
        "filename": image_path,
        "tag": metadata.get("tag", ""),
        "prompt": metadata.get("prompt", ""),
        "correct": correct,
        "reason": "\n".join(reason),
        "metadata": json.dumps(metadata),
        "details": {
            key: [box.tolist() for box, _ in value]
            for key, value in detected.items()
        }
    }
