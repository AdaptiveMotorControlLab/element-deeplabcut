# Element DeepLabCut for Pose Estimation

DataJoint Element for markerless pose estimation with
[DeepLabCut](https://www.deeplabcut.org/).  DataJoint Elements collectively standardize
and automate data collection and analysis for neuroscience experiments.  Each Element is
a modular pipeline for data storage and processing with corresponding database
tables that can be combined with other Elements to assemble a fully functional pipeline.

![flowchart](https://raw.githubusercontent.com/datajoint/element-deeplabcut/main/images/flowchart.svg)

Element DeepLabCut runs DeepLabCut which uses image recognition machine learning models
to generate animal position estimates from consumer grade video equipment.  The Element
is composed of two schemas for storing data and running analysis:

- `train` - Manages model training (for trained models only)
  
- `model` - Manages models and launches pose estimation

The Element supports two workflow modes:

- **Trained Models**: Custom models you train yourself using your own labeled data
- **Pretrained Models**: Ready-to-use models from the DLC Model Zoo (e.g., SuperAnimal models)

Visit the [Workflows page](./workflows.md) to learn about both modes and when to use each.
Visit the [Concepts page](./concepts.md) for more information on pose estimation and
Element DeepLabCut.  To get started with building your data pipeline visit the
[Tutorials page](./tutorials/).
