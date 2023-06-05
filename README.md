This is the repository of **A Survey of Label-Efficient Deep Learning for 3D Point Clouds**, a comprehensive survey of recent progress in deep learning methods for label-efficient learning of point clouds. For details, please refer to: 

 **A Survey of Label-Efficient Deep Learning for 3D Point Clouds**  
 [[Paper](https://arxiv.org/pdf/2305.19812.pdf)] :boom::boom::boom:
 
 [![arXiv](https://img.shields.io/badge/arXiv-2305.19812-b31b1b.svg)](https://arxiv.org/abs/2305.19812)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) 
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
<!-- [![made-with-Markdown](https://img.shields.io/badge/Made%20with-Markdown-1f425f.svg)](http://commonmark.org) -->
<!-- [![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](http://ansicolortags.readthedocs.io/?badge=latest) -->

## Abstract
In the past decade, deep neural networks have achieved significant progress in point cloud learning. However, collecting large-scale precisely-annotated training data is extremely laborious and expensive, which hinders the scalability of existing point cloud datasets and poses a bottleneck for efficient exploration of point cloud data in various tasks and applications. Label-efficient learning offers a promising solution by enabling effective deep network training with much-reduced annotation efforts. This paper presents the first comprehensive survey of label-efficient learning of point clouds. We address three critical questions in this emerging research field: i) the importance and urgency of label-efficient learning in point cloud processing, ii) the subfields it encompasses, and iii) the progress achieved in this area. To achieve this, we propose a taxonomy that organizes label-efficient learning methods based on the data prerequisites provided by different types of labels. We categorize four typical label-efficient learning approaches that significantly reduce point cloud annotation efforts: data augmentation, domain transfer learning, weakly-supervised learning, and pretrained foundation models. For each approach, we outline the problem setup and provide an extensive literature review that showcases relevant progress and challenges. Finally, we share insights into current research challenges and potential future directions. 

<p align="center">
<img src="https://github.com/xiaoaoran/3D_label_efficient_learning/blob/master/asset/Taxonomy_v3.png" width="1024">
</p>

# Citation
```
@article{xiao2023survey,
  title={A Survey of Label-Efficient Deep Learning for 3D Point Clouds},
  author={Xiao, Aoran and Zhang, Xiaoqin and Shao, Ling and Lu, Shijian},
  journal={arXiv preprint arXiv:2305.19812},
  year={2023}
}
```

## Menu
- [Data Augmentation](#data-augmentation)
  - [Intra-domain augmentation](#intra-domain-augmentation)
  - [Inter-domain augmentation](#inter-domain-augmentation)
- [Domain Transfer Learning](#domain-transfer-learning)
  - [Domain Adaptation](#domain-adaptation)
  - [Domain Generalization](#domain-generalization)
- [Weakly-Supervised Learning](#weakly-supervised-learning)
  - [Incomplete Supervision](#incomplete-supervision)
  - [Inexact Supervision](#inexact-supervision)
  - [Inaccurate Supervision](#inaccurate-supervision)
- [Pretrained Foundation Models](#pretrained-foundation-models)
  - [Self-Supervised Pretraining](#self-supervised-pretraining)
  - [Multi-Modal Pretraining](#multi-modal-pretraining)

## Data augmentation
### Intra-domain augmentation
1. Pointaugment: an auto-augmentation framework for point cloud classification. CVPR2020. **[classification]**
2. Pointmixup: Augmentation for point clouds. ECCV2020. **[classification]**
3. Regularization strategy for point cloud via rigidly mixed sample. CVPR2021. **[classification]**
4. Point Cloud Augmentation with Weighted Local Transformations. ICCV2021. **[classification]**
5. PatchAugment: Local Neighborhood Augmentation in Point Cloud Classification. ICCV2021W. **[classification]**
6. PointCutMix: Regularization Strategy for Point Cloud Classification. Neural Computing (2022). **[classification]**
7. Point MixSwap: Attentional Point Cloud Mixing via Swapping Matched Structural Divisions. ECCV2022. **[classification]**
8. SageMix: Saliency-Guided Mixup for Point Clouds. NeurIPS2022. **[classification]**
9. Improving 3D Object Detection through Progressive Population Based Augmentation. ECCV2020. **[detection]**
10. Part-aware data augmentation for 3d object detection in point cloud.. IROS2020. **[detection]**
11. AziNorm: Exploiting the Radial Symmetry of Point Cloud for Azimuth-Normalized 3D Perception. CVPR2022. **[detection]**
12. Mix3D: Out-of-context data augmentation for 3D scenes. 3DV2021. **[semantic segmentation]**
13. PolarMix: A General Data Augmentation Technique for LiDAR Point Clouds. NeurIPS2022.**[semantic segmentation] [detection]**

### Inter-domain augmentation
1. LiDAR-Aug: A General Rendering-based Augmentation Framework for 3D Object Detection. CVPR2021. **[synthetic data] [detection]**
2. Transfer Learning from Synthetic to Real LiDAR Point Cloud for Semantic Segmentation. AAAI2022. **[synthetic data] [semantic segmentation]**
3. Epnet: Enhancing point features with image semantics for 3d object detection. ECCV2020. **[multi-modal] [detection]**
4. PointAugmenting: Cross-Modal Augmentation for 3D Object Detection. CVPR2021. **[multi-modal] [detection]**
5. 2DPASS: 2D Priors Assisted Semantic Segmentation on LiDAR Point Clouds. ECCV2022. **[multi-modal] [semantic segmentation]**

## Domain transfer learning

### Domain Adaptation
1. Pointdan: A multi-scale 3d domain adaption network for point cloud representation. NeurIPS2019. **[classification]**
2. Geometry-aware self-training for unsupervised domain adaptation object point clouds. ICCV2021. **[classification]**
3. RefRec: Pseudo-labels Refinement via Shape Reconstruction for Unsupervised 3D Domain Adaptation. 3DV2021.
4. Self-Supervised Global-Local Structure Modeling for Point Cloud Domain Adaptation with Reliable Voted Pseudo Labels. CVPR2022. **[classification]**
5. Domain adaptation on point clouds via geometry-aware implicits. CVPR2022. **[classification]**
6. Point Cloud Domain Adaptation via Masked Local 3D Structure Prediction. ECCV2022. **[classification]**
7. Quasi-Balanced Self-Training on Noise-Aware Synthesis of Object Point Clouds for Closing Domain Gap. ECCV2022. **[classification]**  
****  
1. Train in Germany, test in the USA: Making 3D object detectors generalize. CVPR2020. **[detection]**
2. SF-UDA3D: Source-Free Unsupervised Domain Adaptation for LiDAR-Based 3D Object Detection. 3DV2020. **[detection]**
3. Adapting Object Detectors with Conditional Domain Normalization. ECCV2020. **[detection]**
4. ST3D: Self-training for Unsupervised Domain Adaptation on 3D Object Detection. CVPR2021. **[detection]**
5. SRDAN: Scale-aware and Range-aware Domain Adaptation Network for Cross-dataset 3D Object Detection. CVPR2021. **[detection]**
6. Unsupervised Domain Adaptive 3D Detection with Multi-Level Consistency. ICCV2021. **[detection]**
7. Fog Simulation on Real LiDAR Point Clouds for 3D Object Detection in Adverse Weather. ICCV2021. **[detection]**
8. SPG: Unsupervised Domain Adaptation for 3D Object Detection via Semantic Point Generation. ICCV2021. **[detection]**
9. Learning Transferable Features for Point Cloud Detection via 3D Contrastive Co-training. NeurIPS2021. **[detection]**
10. LiDAR Snowfall Simulation for Robust 3D Object Detection. CVPR2022. **[detection]**
11. LiDAR Distillation: Bridging the Beam-Induced Domain Gap for 3D Object Detection. ECCV2022. **[detection]**
12. CL3D: Unsupervised Domain Adaptation for Cross-LiDAR 3D Detection. AAAI2023. **[detection]**
****
1. SqueezeSegV2: Improved Model Structure and Unsupervised Domain Adaptation for Road-Object Segmentation from a LiDAR Point Cloud. ICRA2019. **[semantic segmentation]**
2. Domain Transfer for Semantic Segmentation of LiDAR Data using Deep Neural Networks. IROS2020. **[semantic segmentation]**
3. xmuda: Cross-modal unsupervised domain adaptation for 3d semantic segmentation. CVPR2020. **[multi-modal] [semantic segmentation]**
4. ePointDA: An End-to-End Simulation-to-Real Domain Adaptation Framework for LiDAR Point Cloud Segmentation. AAAI2021. **[semantic segmentation]**
5. Complete & label: A domain adaptation approach to semantic segmentation of lidar point clouds. CVPR2021. **[semantic segmentation]**
6. Lidarnet: A boundary-aware domain adaptation model for point cloud semantic segmentation. ICRA2021. **[semantic segmentation]**
7. Sparse-to-dense feature matching: Intra and inter domain cross-modal learning in domain adaptation for 3d semantic segmentation. ICCV2021. **[multi-modal] [semantic segmentation]**
8. MM-TTA: Multi-Modal Test-Time Adaptation for 3D Semantic Segmentation. CVPR2022. **[multi-modal] [test-time adaptation] [semantic segmentation]**
9. Transfer learning from synthetic to real lidar point cloud for semantic segmentation. AAAI2022. **[semantic segmentation]**
10. GIPSO: Geometrically Informed Propagation for Online Adaptation in 3D LiDAR Segmentation. ECCV2022. **[semantic segmentation]**
11. CoSMix: Compositional Semantic Mix for Domain Adaptation in 3D LiDAR Segmentation. ECCV2022. **[semantic segmentation]**
12. DODA: Data-Oriented Sim-to-Real Domain Adaptation for 3D Semantic Segmentation. ECCV2022. **[semantic segmentation]**
13. Adversarially Masking Synthetic to Mimic Real: Adaptive Noise Injection for Point Cloud Segmentation Adaptation. CVPR2023.
14. 3D Semantic Segmentation in the Wild: Learning Generalized Models for Adverse-Condition Point Clouds. CVPR2023. **[semantic segmentation]**
15. SSDA3D: Semi-supervised Domain Adaptation for 3D Object Detection from Point Cloud. AAAI2023. **[semantic segmentation] [Semi-supervised domain adaptation]**

### Domain Generalization
1. MetaSets: Meta-Learning on Point Sets for Generalizable Representations. CVPR2021. **[classification]**
2. Manifold Adversarial Learning for Cross-domain 3D Shape Representation. ECCV2022. **[classification]**
****
1. 3D-VField: Adversarial Augmentation of Point Clouds for Domain Generalization in 3D Object Detection. CVPR2022. **[detection]**
2. Towards Domain Generalization for Multi-View 3D Object Detection in Bird-Eye-View. CVPR2023. **[detection]**
****
1. 3D Semantic Segmentation in the Wild: Learning Generalized Models for Adverse-Condition Point Clouds. CVPR2023. **[semantic segmentation]**
2. Single Domain Generalization for LiDAR Semantic Segmentation. CVPR2023. **[semantic segmentation]**

## Weakly-Supervised Learning
### Incomplete Supervision
1. Weakly supervised semantic point cloud segmentation: Towards 10x fewer labels. CVPR2020. **[semantic segmentation] [3d-weakly]**
2One thing one click: A self-training approach for weakly supervised 3d semantic segmentation. CVPR2021. **[semantic segmentation] [3d-weakly]**
3. Exploring data-efficient 3D scene understanding with contrastive scene contexts. CVPR2021. **[semantic segmentation] [3d-weakly]**
4. Weakly supervised semantic segmentation for large-scale point cloud. AAAI2021. **[semantic segmentation] [3d-weakly]**
5. Learning with Noisy Labels for Robust Point Cloud Segmentation. ICCV2021. **[semantic segmentation] [3d-weakly]**
6. HybridCR: Weakly-Supervised 3D Point Cloud Semantic Segmentation via Hybrid Contrastive Regularization. CVPR2022. **[semantic segmentation] [3d-weakly]**
7. Weakly Supervised Segmentation on Outdoor 4D point clouds with Temporal Matching and Spatial Graph Propagation. CVPR2022. **[semantic segmentation] [3d-weakly]**
8. SQN: Weakly-Supervised Semantic Segmentation of Large-Scale 3D Point Clouds. ECCV2022. **[semantic segmentation] [3d-weakly]**
9. Dual Adaptive Transformations for Weakly Supervised Point Cloud Segmentation. ECCV2022. **[semantic segmentation] [3d-weakly]**
10. Weakly Supervised 3D Scene Segmentation with Region-Level Boundary Awareness and Instance Discrimination. ECCV2022. **[semantic segmentation] [3d-weakly]**
11. LESS: Label-Efficient Semantic Segmentation for LiDAR Point Clouds. ECCV2022. **[semantic segmentation] [3d-weakly]**
12. LiDAL: Inter-frame Uncertainty Based Active Learning for 3D LiDAR Semantic Segmentation. ECCV2022. **[semantic segmentation] [3d-weakly]**
13. Guided Point Contrastive Learning for Semi-supervised Point Cloud Semantic Segmentation. ICCV2021. **[semantic segmentation] [3d-semi]**
14. SSPC-Net: Semi-supervised semantic 3D point cloud segmentation network. AAAI2021. **[semantic segmentation] [3d-semi]**
15. LaserMix for Semi-Supervised LiDAR Semantic Segmentation. CVPR2023. **[semantic segmentation] [3d-semi]**
16. Less is More: Reducing Task and Model Complexity for 3D Point Cloud Semantic Segmentation. CVPR2023. **[semantic segmentation] [3d-semi]**
17. Few-shot 3d point cloud semantic segmentation. CVPR2021. **[semantic segmentation] [few-shot]**
18. Crossmodal few-shot 3d point cloud semantic segmentation. ACM-MM2022. **[semantic segmentation] [few-shot] [multi-modal]**
19. 
****
1. TWIST: Two-Way Inter-label Self-Training for Semi-supervised 3D Instance Segmentation. CVPR2022. **[instance segmentation] [3d-semi]**
2. Geodesic-former: A geodesic-guided few-shot 3d point cloud instance segmenter. ECCV2022. **[instance segmentation] [few-shot]**
****
1. Transferable Semi-Supervised 3D Object Detection From RGB-D Data. ICCV2019. **[detection] [3d-weakly]**
2. SS3D: Sparsely-Supervised 3D Object Detection from Point Cloud. CVPR2022. **[detection] [3d-weakly]**
3. SESS: Self-Ensembling Semi-Supervised 3D Object Detection. CVPR2020. **[detection] [3d-semi]**
4. 3DIoUMatch: Leveraging IoU Prediction for Semi-Supervised 3D Object Detection. CVPR2021. **[detection] [3d-semi]**
5. Semi-supervised 3D object detection with proficient teachers. ECCV2022. **[detection] [3d-semi]**
6. Hierarchical Supervision and Shuffle Data Augmentation for 3D Semi-Supervised Object Detection. CVPR2023. **[detection] [3d-semi]**
7. Prototypical votenet for few-shot 3d point cloud object detection. NeurIPS2022. **[detection] [few-shot]**


****
1. What makes for effective few-shot point cloud classification. WACV2022. **[classification] [few-shot]**
2. Few-shot class-incremental learning for 3d point cloud objects. ECCV2022. **[classification] [few-shot]**
3. Cross-modality feature fusion network for few-shot 3d point cloud classification. WACV2023. **[classification] [few-shot] [multi-modal]**

### Inexact Supervision
1. Multi-path region mining for weakly supervised 3d semantic segmentation on point clouds. CVPR2020. **[semantic segmentation] [3d-weakly]**
2. 3D Spatial Recognition without Spatially Labeled 3D. CVPR2021. **[semantic segmentation] [detection] [3d-weakly]**
3. Scribble-Supervised LiDAR Semantic Segmentation. CVPR2022. **[semantic segmentation] [3d-weakly]**
****
1. Weakly Supervised 3D Object Detection from Lidar Point Cloud. ECCV2022. **[detection] [3d-weakly]**
2. Back to Reality: Weakly-supervised 3D Object Detection with Shape-guided Label Enhancement. CVPR2022. **[detection] [3d-weakly]**
****
1. Point Cloud Instance Segmentation with Semi-supervised Bounding-Box Mining. TPAMI2021. **[instance segmentation] [3d-weakly]**
2. Box2Mask: Weakly Supervised 3D Semantic Instance Segmentation Using Bounding Boxes. ECCV2022. **[instance segmentation] [3d-weakly]**
3. Weakly supervised 3d object detection from point clouds. ACM-MM2020. **[detection] [3d-weakly]**
4. Eliminating Spatial Ambiguity for Weakly Supervised 3D Object Detection without Spatial Labels. ACM-MM2022. **[detection] [3d-weakly]**
### Inaccurate Supervision
1. Learning with noisy labels for robust point cloud segmentation. ICCV2021. **[semantic segmentation]**

## Pretrained Foundation Models
### Self-Supervised Pretraining
Please refer to [Unsupervised Point Cloud Representation Learning with Deep Neural Networks: A Survey (TPAMI2023)](https://github.com/xiaoaoran/3d_url_survey)
### Multi-Modal Pretraining
1. Pointclip: Point cloud understanding by clip. CVPR2022.
2. Partslip: Low-shot part segmentation for 3d point clouds via pretrained image-language models. CVPR2023.
3. Vlsat: Visual-linguistic semantics assisted training for 3d semantic scene graph prediction in point cloud. CVPR2023.
4. Ulip: Learning a unified representation of language, images, and point clouds for 3d understanding. CVPR2023.
5. Pla: Language-driven open-vocabulary 3d scene understanding. CVPR2023.
6. Clip2: Contrastive language-image-point pretraining from real-world point cloud data. CVPR2023.
7. Open-Vocabulary Point-Cloud Object Detection without 3D Annotation. CVPR2023.
