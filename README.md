# Industrial Surface Defect Segmentation (YOLOv8-seg on MVTec AD)

Instance segmentation of surface defects on pharmaceutical capsules, using Ultralytics YOLOv8 and the MVTec AD dataset. The pipeline converts the dataset's pixel-level ground-truth masks into YOLO polygon labels, trains on a small dataset with augmentation and defect-free negative samples, and evaluates both segmentation quality and false-positive behaviour.

![Sample prediction 1](assets/demo_1.png)
![Sample prediction 2](assets/demo_2.png)

## Results

All numbers are measured on a held-out validation set of 44 images (21 defective images with 23 defect instances, plus 23 defect-free images). Final model: **single-class YOLOv8n-seg**.

| Run | Model | Classes | Mask mAP50 | Mask mAP50-95 | Mask precision | Mask recall | False positives (defect-free) |
|---|---|---|---|---|---|---|---|
| A | YOLOv8n-seg | 5 defect types | 0.560 | 0.287 | 0.606 | 0.502 | 0 / 23 |
| **B (final)** | **YOLOv8n-seg** | **1 (defect)** | **0.664** | **0.295** | **0.874** | **0.602** | **0 / 23** |
| C | YOLOv8s-seg | 1 (defect) | 0.369 | 0.173 | 0.420 | 0.378 | not measured |

False positives are counted as defect-free test images on which the model predicts any mask at `conf=0.25`.

### Per-class results for run A (5 classes, mask mAP50)

| Class | Validation instances | Mask mAP50 |
|---|---|---|
| faulty_imprint | 3 | 0.995 |
| poke | 9 | 0.943 |
| scratch | 7 | 0.514 |
| crack | 3 | 0.321 |
| squeeze | 1 | 0.000 |

Crack and squeeze have only 1 to 3 validation instances, so their scores are very noisy.

## Key findings

- **Merging defect types into one class helped.** With roughly 20 defective images per type, the 5-class setup spread too little data across too many classes. Mask mAP50 rose from 0.56 to 0.66 and precision from 0.61 to 0.87. Note that the single-class task is easier ("where is a defect?" rather than "which type is it?"), so the numbers are not a like-for-like comparison.
- **A larger model did not help.** YOLOv8s-seg scored well below YOLOv8n-seg on this dataset. With about 88 defective training images, the larger model likely overfits or converges poorly. Differences between single runs on a validation set this small are also noisy.
- **Precision is the strength.** Mask precision of 0.87 and zero false alarms on 23 defect-free images suit the goal of not stopping a production line for fake defects.

## Limitations

- Recall is about 0.60, so roughly 4 in 10 defect instances are missed at the default threshold.
- mAP50-95 is around 0.30. Thin cracks and scratches are hard to outline tightly, and this metric rewards tight mask overlap.
- The validation set is small (23 defect instances), so all metrics carry wide uncertainty. Results were not averaged over multiple seeds or splits.
- Only the `capsule` category was used. Results may not transfer to other categories.
- The dataset is MVTec AD, which is licensed for non-commercial use only.

## Method

1. **Labels.** MVTec AD ships binary ground-truth masks for every defective test image. Each mask is converted to polygon contours with OpenCV (`findContours` + `approxPolyDP`) and written in YOLO segmentation format. Tiny contours (area < 20 px) are dropped.
2. **Split.** The `train/` folder of MVTec AD contains only defect-free images, so defective images come from `test/`. They are shuffled with a fixed seed (42) and split 80/20 into train and validation (88 train, 21 validation).
3. **Negative samples.** 100 defect-free images from `train/good` are added to training with empty label files. All 23 images from `test/good` are used in validation only, so the false-positive check never overlaps with training.
4. **Augmentation.** Ultralytics built-in augmentation: brightness and saturation shifts (`hsv_v=0.4`, `hsv_s=0.4`), rotation (15 degrees), translation, scaling, vertical and horizontal flips, and mosaic.
5. **Training.** 100 epochs maximum, early stopping (`patience=20`), image size 800, batch size 8, on a Colab T4 GPU.

## How to reproduce

1. Download the **Capsule** category from the [MVTec AD page](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract it so that you have `mvtec_ad/capsule/{train,test,ground_truth}`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `notebook.ipynb` (best run on Google Colab with a GPU) and run the cells in order:
   - build the YOLO dataset from the masks
   - train
   - evaluate (`model.val`)
   - false-positive check
   - demo predictions
4. Adjust the `SRC` and `DST` paths at the top of the notebook if you are not using Colab.

Trained weights for the final model are in `weights/best.pt`.

## Project structure

```
.
├── notebook.ipynb      # full pipeline: data prep, training, evaluation
├── weights/best.pt     # final single-class YOLOv8n-seg model
├── assets/             # sample prediction images
├── requirements.txt
└── README.md
```

## Tech stack

Python, PyTorch, Ultralytics YOLOv8 (tested with 8.4.176), OpenCV, Google Colab.

## Acknowledgements

Dataset: Bergmann et al., *MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019. Not redistributed in this repository.
