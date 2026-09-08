# Literature Notes

Short summaries of the papers that informed the design of this pipeline.

---

## 1. Rezaei et al. (2026)  
**Automatic pixel-level annotation for plant disease severity estimation**  
*Computers and Electronics in Agriculture*

Weakly-supervised pipeline for barley net blotch severity.  
ViT + GradCAM generates pixel-level pseudo-labels → lightweight SOD models (MEANet, SeaNet) for lesion/leaf segmentation → ABCD + GLCM features → SVM / feature-fusion network for severity classification.  
Cross-dataset tests on Coffee, Rice, Wheat. MEANet + SVM/FF5 reach ~82 %+ accuracy on Coffee.

**Relevance:** Reinforces the *segment-then-quantify* paradigm and the value of explicit leaf vs lesion masks.

---

## 2. Raj, Prakash & Malik — AgriFusionNet  
*Procedia Computer Science*

Multi-task CNN (Res2Net + dilated conv + ECA + dense connections, MSFFM) for joint disease classification (25 classes) and 5-level severity grading.  
Claims 98.24 % accuracy, outperforming ResNet50, VGG19, DenseNet121, EfficientNetB0, MobileNetV2.

**Relevance:** Shows that multi-task (classification + severity) heads are viable; our pipeline keeps severity as a pure geometric measurement instead.

---

## 3. Wu et al. — AgriMamba  
*The Crop Journal*

Three-stage Mamba/SSM multimodal framework:  
1. Localization-aware mamba leaf segmenter (LMLS)  
2. Text-guided mamba lesion segmenter (TMLS, CLIP + SGMF)  
3. Area-ratio severity grading (6 levels)  

New PlantMM-SG dataset (10 350 samples, 16 crops). 97.20 % grading accuracy; beats several strong segmentation baselines and large VLMs.

**Relevance:** Direct inspiration for **dual-stage refinement** (leaf → crop → lesion) and for treating severity as an area ratio.

---

## 4. Zhang et al. — RSD-YOLO  
*Smart Agricultural Technology*

Improved YOLOv7-tiny for oat disease severity (5 classes: H / R / MR / MS / S).  
ReXNet backbone, Slim-Neck, Decoupled Head. New 1 010-image oat dataset.  
91.6 % P / 90.8 % R / 88.5 % mAP@0.5 at 11.2 GFLOPs.

**Relevance:** Confirms that lightweight YOLO-family detectors are practical for field severity work; we adopt YOLOv8-seg for the same reason.

---

## 5. Shivaraj & Haladappa  
**A Dual-Stage Deep Learning Framework for Phyllosticta Leaf Disease Severity Estimation in Cardamom**  
*ETASR*

Two-stage U-Net:  
1. Isolate leaf  
2. Segment lesions on the cropped leaf (BCE + Dice loss)  

Severity = lesion area / leaf area × 100 → four grades (Healthy / Low / Medium / High).  
Leaf Dice 0.982, lesion Dice 0.914, severity MAE 1.87 %, overall accuracy 94.2 %.  
Ablation clearly favours dual-stage over single U-Net.

**Relevance:** Primary source for the **ENABLE_DUAL_STAGE_REFINEMENT** option in this pipeline.

---

## 6. Astani & Hasheminejad — BaSPaC  
**A Parallel Convolutional Neural Network with Background Removal and Lesion Segmentation for Field Plant Disease Severity Classification**  
*Scientific Reports*

Focus on tomato Bacterial Spot and Mosaic Virus under domain shift (lab ↔ field).  
Key finding: cross-domain accuracy collapses to ~33–47 %; background/shadow removal + segmentation recovers a large part of it.  
Proposes BaSPaC: Cascaded MRCNN (leaf + lesion) + AISA (hyper-green + GrabCut) + parallel CNN branches.

**Relevance:** Motivates **ENABLE_BG_SHADOW_REMOVAL** and the emphasis on field-robust preprocessing.

---

## 7. Chiang & Hong  
**Advances in Optimizing Quantitative Ordinal Scales and Statistical Analysis Methods for Plant Disease Severity**  
*Journal of Plant Medicine* (in Chinese)

Methodological review (no deep learning).  
- Critiques the Horsfall–Barratt scale (wide mid-range bins reduce statistical power).  
- Proposes the **Chiang scale**: fine intervals below 10 %, then simple 10 % bins.  
- Discusses Disease Severity Index (DSI) bias when ordinal grades are treated as linear.  
- Recommends proportional-odds models or interval-censored survival analysis for ordinal severity data.

**Relevance:** Default severity scale in this repo is `"chiang"`. The paper also warns against naïve DSI formulas—our pipeline reports continuous % plus an ordinal grade rather than a single biased index.

---

## Design takeaways used in this repository

| Idea | Source | Implementation |
|------|--------|----------------|
| Leaf-then-lesion dual stage | Shivaraj & Haladappa, AgriMamba | `ENABLE_DUAL_STAGE_REFINEMENT` |
| Background / shadow handling | BaSPaC | `ENABLE_BG_SHADOW_REMOVAL` |
| Severity as area ratio | Multiple | `src/models/severity.py` |
| Chiang ordinal scale | Chiang & Hong | `SEVERITY_SCALE: chiang` |
| Lightweight YOLO backbone | RSD-YOLO family | `yolov8n-seg.pt` default |
| VLM as advisory critic | Modern multimodal work | Qwen2.5-VL verification layer |

---

## Citation reminder

When publishing results that use the Roboflow dataset, also cite:

```bibtex
@misc{leaf-lesion-segmentation-qfbaa_dataset,
  title        = {leaf lesion segmentation Dataset},
  type         = {Open Source Dataset},
  author       = {balajiCV},
  howpublished = {\url{https://universe.roboflow.com/balajicv/leaf-lesion-segmentation-qfbaa}},
  journal      = {Roboflow Universe},
  publisher    = {Roboflow},
  year         = {2025},
  month        = {nov}
}
```
