# reg-nf-dev-temp
Temporary private repo for tidying Reg-NF code before public release in Reg-NF repo

# Install
The code here is designed to work within a pre-existing sdfstudio repository. Please see the [sdfstudio repository](https://github.com/autonomousvision/sdfstudio) to install sdfstudio.

Once sdfstudio is installed, simply copy the relevant contents of this repo into sdfstudio, install remainging dependancies and update the package.

```
git clone <TODO regnf repo>
cp Reg-NF/scripts <path/to/sdfstudio>/
cp Reg-NF/configs <path/to/sdfstudio>/
cd <path/to/sdfstudio>
pip install visdom git+https://github.com/jonbarron/robust_loss_pytorch
pip install -e .
```
Now you should be ready to use RegNF

# Using RegNF
RegNF can be broken up into 3 steps with three scripts. Training SDF models, registration through reg-nf, and rendering/evaluating results.

## Training SDF models
SDF models are trained using existing sdfstudio procedures with some notable assumptions being made for interaction with the rest of the code base.

1. Training is done using the `neus-facto` model
2. Naming convention for experiments is `neus-facto-<name>-<suffix>-<img_size>` where `<name>` is the name of the object/scene being described, `<suffixes>` are any suffixes listed in the rest of the codebase (e.g. "short" for short trajectories in scenes) and `<img_size>` is a single integer indicating the size of the image used in training.
3. Data used for training is in the same format as the [ONR dataset](https://doi.org/10.25919/0vbj-fk61) (containing model GT names and poses for evaluation). For more info see ReadMe in the link above.

**Example object training (w. depth data)**
```
ns-train neus-facto \
    --pipeline.model.sdf-field.inside-outside False \
    --pipeline.model.sdf-field.geometric-init True \
    --pipeline.model.sdf-field.bias 0.5 \
    --pipeline.model.near-plane 0.05 \
    --pipeline.model.far-plane 2.5 \
    --pipeline.model.overwrite-near-far-plane True \
    --pipeline.model.background-model none \
    --pipeline.model.background-color white \
    --pipeline.model.sensor-depth-l1-loss-mult 0.1 \
    --pipeline.model.sensor-depth-sdf-loss-mult 6000 \
    --pipeline.model.sensor-depth-freespace-loss-mult 10.0 \
    --pipeline.model.sensor-depth-truncation 0.0248 \
    --trainer.steps-per-eval-image 1000 \
    --trainer.steps-per-eval-batch 1000 \
    --trainer.steps-per-save 1000 \
    --trainer.max-num-iterations 30001 \
    --experiment-name neus-facto-fc \
    sdfstudio-data \
    --data data/omniverse/objects/fancy_chair_512_rtxi/ \
    --include-sensor-depth True
```

Once you have at least one scene model and one object model, you are ready to perform RegNF

## RegNF Registration
Once you have a scene and an object SDF model as outlined above, you are ready to use RegNF to perform registration. 

You do this for a single object with it's matching object within a scene without any visualisations using `regnf.py` as follows:

```
python scripts/regnf.py --scene-sdf-path <path/to/scene.yml> --object-sdf-path <path/to/object.yml> --transform-savepath <path/to/transform.npy> sdfstudio-data <path/to/scene/data>
```

Options exist to perform visualisations with visdom or open3d, saving open3d visualisations of each step of optimization, saving the initialisation, FGR or each step transformation matrices, as well as to enable the provided object to be tested against every object within the scene. For further details run `python scripts/regnf.py -h`.

**Note** the savepath provided in `--transform-savepath` will have an additional suffix added to the end of it, indicating which object within the scene the provided object was matched with (useful for when testing against all objects within the scene)

**WARNING** To save different transform outputs and use these results in later evaluation scripts, we assume the following naming conventions

1. "transforms" should be included as a folder along the savepath
2. Final file follows the format `<scene_model>-2-<object_model>.npy` where  both `<scene_model>` and `object_model` are short names included both in the `PRIM_TO_OBJECT_SHORT` and `OBJECT_FULLNAMES` dictionaries found in `regnf.py`

## Evaluating RegNF Results
If transform matrices and scene and object model files have been set up according to the naming conventions outlined above, the transformation numpy file will be all that is needed to perform evaluation as follows:

```
python scripts/evaluate_transforms.py --tmat_file <path/to/transform.npy> --result_csv <path/to/csv>
```

Where the csv file provided will have the latest evaluation results appended to it. 

**Note** there is also the option to add a `--run_id` to this script to indicate multiple runs of the same experiment setup.

## RegNF Scene Editing Rendering with RegNF Results
As shown in the paper, RegNF results can enable library substitution or object instance replacement within the scene.


# Set up ONR dataset

## Known issues
We have found there are some issues using the pre-trained models in the ONR dataset with different versions of sdfstudio as they had been trained on different versions.

**Working with latest sdfstudio**
* chair
* dining_chair
* matrix_chair

**working on older sdfstudio**
* fancy_chair
* fancy_chair_no_pillow
* end_table
* table
* willow_table

To use the older models run `pip install torchmetrics==0.11.4 lpips`. To revert back run `pip install torchmetrics==1.4.0`. For simplicity we recommend training new models from scratch with the data provided.
