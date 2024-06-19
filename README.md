# **Oral3**


This GitHub repo for the design analysis of different Case-Based Reasoning (CBR) systems enhanced by Deep Learning (DL) to support oral cancer diagnosis.

<img width="560" alt="intro" src="img/DL-CBR-scheme.drawio.png">

Two main DL strategies have been adopted for DL module design to get a feature extractor to be integrated in DL-CBR system:
- Supervised Learning (SL)
- Self-supervised Learning (SSL)

Each of these strategies can rely on different loss function during the training. The following table present the SOTA architectures adopted in this project

| Model      | DL strategy | loss function |
|------------|-------------|---------------|
| ConvNext   | SL          |               |
| SqueezeNet | SL          |               |
| Vit_b      |  SL         |               |
| Swin S     |   SL        |               |
| TripletNet | SL          |               |
| Conv. AutoEncoder | SSL          |               |
| Dino      |  SSL         |               |
| VitMAE     |   SSL        |               |




Additional information and instuction for the usage can be found in the [official documentation](docs/README.md)