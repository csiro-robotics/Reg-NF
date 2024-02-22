## Reg-NF: Efficient Registration of Implicit Surfaces within Neural Fields


### Contributions
* TODO

### Abstract

Neural fields, coordinate-based neural networks, have recently gained popularity for implicitly representing a scene. In contrast to classical methods that are based on explicit representations such as point clouds, neural fields provide a continuous scene representation able to represent 3D geometry and appearance in a way which is compact and ideal for robotics applications. However, limited prior methods have investigated registering multiple neural fields by directly utilising these continuous implicit representations. In this paper, we present Reg-NF, a neural fields-based registration that optimises for the relative 6-DoF transformation between two arbitrary neural fields, even if those two fields have different scale factors. Key components of Reg-NF include a bidirectional registration loss, multi-view surface sampling, and utilisation of volumetric signed distance functions (SDFs). We showcase our approach on a new neural field dataset for evaluating registration problems. We provide an exhaustive set of experiments and ablation studies to identify the performance of our approach, while also discussing limitations to provide future direction to the research community on open challenges in utilizing neural fields in unconstrained environments.


### Visual Summary

HERO VIDEO GOES HERE. Triple video, first is alignment process, second is substit, third is replacement.

<p align="center">
<img src="assets/imgs/hero_img_designer.jpg" style="width:70%">
</p>


### Method
Short summary of total method. Hero method pic.
<p align="center">
<img src="assets/imgs/optimization_img.png" style="width:90%">
</p>

#### Novel Bi-directional Optimisation 
After a BLAH initialisaiton step, we introduce our bi-directional optimisation procedure. INFO DUMP

Through our experiments we show that our registration process outperformed the previous state-of-the-art in neural field registration as seen below.


### Results
Reg_NF is able to cope with registration of objects at different scale.

VIDEO OF NERF2NERF VS REG-NF

#### Use-cases - Library model substitution
After registration, library models can be substituted into the scene to help handle low-coverage or under-trained scenes.

Video of low-coverage model comparison

Video of under-trained model comparison

#### Use-cases - Library model replacement

#### ONR Dataset



### Download

The dataset will be released very soon.


### Code
 <p>
    We provide code for running our neural field registration code <a href="https://github.com/csiro-robotics/Reg-NF">in our GitHub repository</a> as well as code for utilizing our ONR dataset of simulated environments.
</p>

### Paper

The pre-print version of the paper is available on arxiv at: .

### Citation
<p>
If you find this paper helpful for your research, please cite our paper using the following reference:

```
@misc{hausler2024regnf,
      title={RegNF: Efficient Registration of Implicit Surfaces within Neural Fields}, 
      author={Stephen Hausler and David Hall and Sutharsan Mahendren and Peyman Moghadam},
      year={2024},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
</p>