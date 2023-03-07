<div align="center">   
  
# Toward Zero-Shot Sim-to-Real Transfer Learning for Pneumatic Soft Robot 3D Proprioceptive Sensing
</div>


# Abstract
Pneumatic soft robots present many advantages in
manipulation tasks. Notably, their inherent compliance makes
them safe and reliable in unstructured and fragile environ-
ments. However, full body shape sensing for pneumatic soft
robots is a difficult challenge because of their high degrees
of freedom and complex deformation behaviors. Vision-based
proprioception sensing methods relying on embedded cameras
and deep learning provide a good solution to proprioception
sensing by extracting the full-body shape information from the
high-dimensional sensing data. But the current training data
collection process makes it difficult for many applications. To
address this challenge, we propose and demonstrate a robust
sim-to-real pipeline that allows the collection of the soft robot’s
shape information in high-fidelity point cloud representation.
The model trained on simulated data was evaluated with
real internal camera images. The results show that the model
performed with averaged Chamfer distance of 8.85 mm and
tip position error of 10.12 mm even with external perturbation
for a pneumatic soft robot with a length of 100.0 mm. We also
demonstrated the sim-to-real pipeline’s potential for exploring
different configurations of visual patterns to improve vision-
based reconstruction results. The code and dataset are available https://github.com/DeepSoRo/DeepSoRoSim2Real.


# Method

| ![space-1.jpg](results/front.png) | 
|:--:| 
| ***Figure 1. Overall framework**. The proposed pipeline of sim-to-real transfer learning for vision-based soft robot. We generated simulation-based point cloud and corresponding internal camera views to train our neural network model. Then, we show that the trained model transfers zero-shot to the real world by testing with real-world images..* |


# Dataset 
Download link (upcoming ...)

# Bibtex
If this work is helpful for your research, please cite the following BibTeX entry.

```

```

# Acknowledgement