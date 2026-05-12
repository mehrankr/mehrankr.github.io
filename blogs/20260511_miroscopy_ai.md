---
author: Mehran Karimzadeh
date: May 11, 2026
description: A historical review of AI for microscopy image analysis, from classical thresholding to foundation models, spatial proteomics, and multi-modal integration.
---

# AI for Microscopy Image Analysis: A Historical Review

**From Classical Thresholding to Foundation Models, Spatial Proteomics, and Multi-Modal Integration (2000–2024)**

---

## Contents

1. [Why Microscopy Images Need AI](#why-microscopy-images-need-ai)
2. [Types of Microscopy Images](#types-of-microscopy-images)
3. [A Timeline of the Field](#a-timeline-of-the-field)
4. [Era 1, Classical Computer Vision](#era-1--classical-computer-vision-2000--2012)
5. [Era 2, Early Deep CNNs](#era-2--early-deep-convolutional-networks-2012--2015)
6. [Era 3, Semantic Segmentation & U-Net](#era-3--semantic-segmentation--u-net-2015--2018)
7. [Era 4, Instance Segmentation](#era-4--instance-segmentation-2018--2021)
8. [Era 5, Vision Transformers](#era-5--vision-transformers-2020--2022)
9. [Era 6, Self-Supervised & Foundation Models](#era-6--self-supervised-learning--foundation-models-2022--2024)
10. [Application: CODEX & Multiplexed Protein Imaging](#codex--multiplexed-protein-imaging)
11. [Application: H&E Histopathology](#he-histopathology-from-whole-slide-images-to-clinical-prediction)
12. [Open Challenges](#open-challenges)
13. [Summary: Major Advances and Remaining Challenges](#summary-major-advances-and-remaining-challenges)
14. [Summary Table](#summary-methods-across-the-field)
15. [References](#references)

---

## Motivation

For the past three decades, high-throughput biology has been dominated by molecular profiling, specifically through sequencing RNA and DNA, and measuring proteins via tandem mass spectrometry. But while we have become incredibly adept at tracking nanoscale changes in a cell's smallest molecules, figuring out which of those changes actually matter often requires zooming out. To truly understand cellular function, we need macro-scale context as much as micro-scale context. Thanks to new spatial omics techniques that combine high-throughput data with high-resolution imaging, microscopy has reclaimed its role as a vital tool for exploring cells and tissues. This shift inspired me to put together this review. Please reach out if you notice any important studies missing from this post or if you have suggestions for improvement!

## Why Microscopy Images Need AI

Microscopy is the primary language through which cell biology speaks. Every cell-division event, every tumor invasion front, every cellular contact, these phenomena are ultimately read from images.

A single whole-slide histopathology scan is a gigapixel image, roughly the resolution of 10,000 smartphone photos stitched together. A CODEX experiment measuring 40 protein markers across a tissue section generates hundreds of raw fluorescence images per sample. A high-content screening run testing drug effects on cultured cells might image a million cells per day. Similarly, 10x Xenium can profile RNA readout from up to 5,000 genes per cell in a histopathologic slide, generating millions of reads from thousands of coordinates. No team of human annotators can keep pace.


## Types of Microscopy Images

Different imaging modalities produce fundamentally different data structures, noise profiles, and biological questions. Understanding these distinctions is essential for choosing the right AI approach.

### Brightfield & H&E Histopathology

Hematoxylin and Eosin (H&E) staining is the gold standard of clinical pathology. Hematoxylin binds basophilic structures, most prominently cell nuclei, staining them blue-purple. Eosin counterstains eosinophilic cytoplasmic structures pink. The result is a 24-bit RGB image familiar to any pathologist. Modern whole-slide scanners produce images of 40,000 × 40,000 pixels or more at 40× objective magnification. Key AI tasks include: tumor detection and grading, mitotic figure counting, survival prediction, and mutation status inference from morphology alone.

### Widefield and Confocal Fluorescence Microscopy

Fluorescence microscopy labels specific cellular components with fluorescent dyes or genetically encoded proteins (e.g., GFP). Each fluorescent channel is a separate grayscale image; a typical experiment produces 3–7 channels (e.g., DAPI for nuclei, a cytoplasmic marker, and one or more protein-of-interest channels). Confocal microscopes add optical sectioning: a pinhole rejects out-of-focus light, enabling 3D reconstructions as stacks of z-slices. Noise in fluorescence images follows a Poisson-Gaussian mixture: low photon counts produce shot noise, while the detector adds additive Gaussian read-out noise.

### Multiplexed Protein Imaging: CODEX and Related Platforms

CO-Detection by indEXing (CODEX; Goltsev et al., 2018) and related platforms, Imaging Mass Cytometry (IMC), Multiplexed Ion Beam Imaging (MIBI), and CyCIF, enable measurement of 40–80 protein markers in a single tissue section at single-cell spatial resolution. In CODEX, antibodies are conjugated to short DNA barcodes. Imaging proceeds in iterative cycles: a subset of barcodes is hybridized with fluorescent reporters, imaged, reporters are stripped, and the cycle repeats. The result is a multichannel image with one channel per protein. This volume of protein information, resolved spatially within intact tissue architecture, demands AI for cell segmentation, signal normalization, cell-type classification, and spatial neighborhood analysis.

### Spatial Transcriptomics: 10x Visium

Spatial transcriptomics bridges the gap between histology and genomics by mapping RNA expression directly onto tissue architecture. In the 10x Visium platform, intact tissue sections are placed onto slides arrayed with millions of spatially barcoded capture probes. The resulting dataset is inherently multimodal: it pairs a high-resolution tissue image (typically H&E or multiplexed immunofluorescence) with a highly sparse, high-dimensional gene expression matrix mapping the transcriptome to specific spatial coordinates. Because standard Visium spots are roughly 55 µm in diameter, they capture the aggregated RNA of multiple cells (typically 1–10) rather than true single cells. Key AI tasks include multimodal data integration, spatial domain clustering, spot deconvolution (inferring the mixture of single-cell types within a given spot), and leveraging deep learning to impute missing spatial gene expression directly from histological morphology.

### Electron Microscopy (Cryo-EM and Volume EM)

Transmission electron microscopy reveals subcellular structure at nanometer resolution. Single-particle cryo-EM determines protein structures from thousands of noisy 2D projection images of identical molecules. Volume EM (FIB-SEM, serial section TEM) generates 3D volumes of cellular ultrastructure, organelles, synapses, vesicles, at 5–10 nm voxel resolution. Image segmentation in these data sets is among the most challenging in biology: image contrast is inverted compared to light microscopy, noise is substantial, and objects of interest (mitochondria, synaptic vesicles) are densely packed and irregular.

### Super-Resolution Microscopy

STORM, PALM, STED, and structured illumination microscopy (SIM) bypass the diffraction limit (~200 nm) to image molecular-scale structures. Single-molecule localization methods (STORM, PALM) require fitting point-spread functions to millions of blinking fluorophore events to reconstruct a super-resolution image. AI methods, particularly convolutional neural networks, now replace classical fitting algorithms for localization, and generative models perform deep-learning-based super-resolution from diffraction-limited inputs.

> **Key Insight:** Each modality defines its own "grammar" of image formation: noise model, spatial resolution, channel semantics, and object geometry. The most effective AI methods are those designed with awareness of these constraints, not blind application of a generic computer vision architecture.

---

## A Timeline of the Field

| Year | Event |
|------|-------|
| 2006 | CellProfiler (Carpenter et al.), first open-source, systematic cell-image analysis platform |
| 2012 | AlexNet wins ImageNet; deep CNNs enter biology within months |
| 2015 | U-Net (Ronneberger et al.), encoder-decoder with skip connections; defines biomedical segmentation |
| 2018 | CODEX (Goltsev et al.), 28-marker spatial protein imaging of mouse spleen; StarDist published |
| 2018 | CARE (Weigert et al.), content-aware restoration of fluorescence images with U-Net |
| 2019 | DeepCell / Mesmer (Van Valen lab), whole-cell segmentation from nuclear + membrane channels |
| 2020 | ViT (Dosovitskiy et al.), Vision Transformer; patches replace pixels as tokens |
| 2021 | Cellpose (Stringer et al.), gradient-flow segmentation; Swin Transformer (Liu et al.) |
| 2021 | CLAM (Lu et al.), attention-based MIL for gigapixel slide classification |
| 2022 | MAE (He et al.); Noise2Void; SAM enters development; Cellpose 2.0 |
| 2023 | SAM (Kirillov et al.), Segment Anything; PLIP pathology vision-language model |
| 2024 | UNI, CONCH, Prov-GigaPath, gigascale pathology foundation models; SAM 2 |

---

## Era 1, Classical Computer Vision (2000 – 2012)

*Before deep learning, cell image analysis relied on hand-crafted features and deterministic algorithms. Powerful for constrained settings; brittle when biological variability increased.*

### Thresholding and Morphological Operations

The oldest approach to segmenting cells from microscopy images is **global or local thresholding**: declare pixels above an intensity threshold as foreground (cell), below as background. Otsu's method (1979) formalizes this by finding the threshold that minimizes within-class intensity variance. It works well when the intensity histogram has two clearly separated peaks (bimodal). **Limitation**: real fluorescence images exhibit uneven illumination (vignetting), out-of-focus haze, and widely variable staining intensities, all of which collapse the bimodal assumption.

**Morphological operations** (erosion, dilation, opening, closing) and the **watershed transform** were layered on top of thresholding to separate touching cells. The watershed algorithm treats a grayscale image as a topographic relief map and "floods" basins from local intensity minima. Each basin becomes a cell instance. **Limitation** of watershed: it is exquisitely sensitive to noise (spurious local minima = over-segmentation) and fails entirely when cells have elongated or concave shapes, or when cell boundaries have lower contrast than cell interiors.

### CellProfiler (2006)

The Broad Institute's CellProfiler (Carpenter et al., 2006) was a landmark: an open-source pipeline system that codified the best practices of classical image analysis into a modular, reproducible workflow. Users chain modules for illumination correction, thresholding, object detection, and feature extraction. CellProfiler computes hundreds of handcrafted morphological, intensity, and texture features per segmented cell, which downstream classifiers (random forests, SVMs) use for phenotyping.

CellProfiler's **limitation**: every pipeline must be manually designed and tuned for each assay. Features engineered for one cell type generalize poorly to another. When crowded 3D tissues replaced flat cell monolayers, classical pipelines degraded rapidly.

**Strengths of Classical CV:**
- Interpretable, every decision traceable
- No training data required
- Fast on standard hardware
- Reproducible given same parameters

**Limitations:**
- Fragile under staining / illumination variation
- Cannot separate touching / overlapping cells reliably
- Feature engineering is assay-specific
- 3D segmentation is prohibitively complex to engineer manually

---

## Era 2, Early Deep Convolutional Networks (2012 – 2015)

*AlexNet's 2012 ImageNet victory demonstrated that CNNs could learn image representations vastly superior to hand-crafted features. Biology rapidly adopted them, first for classification, then for patch-level detection.*

### The Convolutional Layer

A convolutional layer applies a bank of learned filters to an input feature map. For a 2D image $\mathbf{I}$ and a filter $\mathbf{W}$ of spatial size $k \times k$, the output activation at position $(i,j)$ is:

$$
z_{i,j} = \sigma\!\left(\sum_{m=0}^{k-1}\sum_{n=0}^{k-1} W_{m,n} \cdot I_{i+m,\; j+n} \;+\; b\right)
$$

where $\sigma$ is a nonlinear activation function (typically ReLU: $\sigma(x)=\max(0,x)$), and $b$ is the bias. Each filter slides across the full spatial extent of $\mathbf{I}$, sharing weights everywhere, giving translation equivariance and drastically fewer parameters than a fully connected layer.

Stacking multiple convolutional layers with interleaved **max-pooling** (taking the maximum value within a local window) builds a hierarchy: early layers detect edges and blobs; deeper layers combine these into nuclei, cell boundaries, and eventually whole-cell shapes. **Max-pooling** halves the spatial resolution at each stage, expanding the receptive field but losing precise localization.

### Application to Microscopy: Classification and Patch Detection

The earliest microscopy deep-learning papers (2013–2015) applied AlexNet-style networks to **patch classification**: slide a window across a microscopy image, classify each patch as cell/no-cell, cancerous/normal, or a cell-type label. This approach inherited ImageNet's classification head (global average pooling or fully connected layers) and output a single label per patch rather than a per-pixel segmentation. **Limitation**: classification gives no spatial precision. To segment cells, to know exactly which pixels belong to which cell, the network must produce a dense, pixel-wise output. Standard CNNs ending in fully connected layers discard all spatial information in the final layers.

> **Key Insight:** The fundamental architectural incompatibility between classification CNNs (global output) and segmentation (dense spatial output) was the central challenge the next era resolved.

---

## Era 3, Semantic Segmentation & U-Net (2015 – 2018)

*Replacing fully connected layers with upsampling paths restored spatial resolution. Skip connections fused high-resolution encoder features with semantically rich decoder features, enabling pixel-precise cell segmentation for the first time.*

### Fully Convolutional Networks (FCN, 2015)

Long, Shelhamer, and Darrell (2015) made a simple but profound change: replace the fully connected layers at the end of AlexNet and VGG with $1\times1$ convolutional layers, preserving spatial dimensions. They then **upsample** the low-resolution feature map back to input size via learned transposed convolutions (sometimes called deconvolutions). The result is a dense prediction map assigning a class label to every pixel, *semantic segmentation*.

**Limitation** of FCN: upsampling from heavily downsampled feature maps (stride 32 in VGG) loses fine spatial detail. Predicted boundaries are blurry. The network knows *where approximately* a nucleus is, but not its precise edge, critical for accurate cell-by-cell quantification.

### U-Net (Ronneberger, Fischer & Brox, 2015), MICCAI 2015

U-Net, designed specifically for biomedical image segmentation, solved FCN's boundary-blurring problem with **skip connections**. The architecture is symmetric: an *encoder* path progressively halves spatial resolution (like standard CNN classification); a *decoder* path progressively restores it via transposed convolutions. Crucially, at each resolution level, the corresponding encoder feature map is **concatenated** to the decoder feature map, directly passing high-resolution spatial detail forward around the bottleneck.

**U-Net Architecture:**

```
Encoder block at level ℓ:
  Conv → BN → ReLU → Conv → BN → ReLU → MaxPool(2×2)
  Skip connection: Feature map F_enc^(ℓ) saved before pooling.

Decoder block at level ℓ:
  TransposedConv(2×2) → Concat[F_enc^(ℓ), upsampled] → Conv → BN → ReLU → Conv → BN → ReLU

Final layer:
  Conv(1×1) → Sigmoid (binary) or Softmax (multiclass) → per-pixel prediction map
```

The concatenation is the key operation. Encoder features carry precise spatial coordinates but lack semantic context (they are "early" features). Decoder features carry semantic understanding but blurry locations (they are "deep" features after many poolings). Concatenating them lets the decoder answer: "I know there should be a nucleus boundary here (semantic, from decoder) and it is precisely these pixels (spatial, from encoder skip)."

### Loss Functions in U-Net

Ronneberger et al. used a class-weighted binary cross-entropy loss, with a **weight map** $w(\mathbf{x})$ that boosted the loss at pixels near touching-cell boundaries. Let $\Omega$ denote the set of all pixels in the image and $|\Omega|$ its cardinality (total pixel count). Defining $p(\mathbf{x}) \in [0,1]$ as the predicted probability that pixel $\mathbf{x} \in \Omega$ belongs to the foreground (cell interior) and $y(\mathbf{x}) \in \{0,1\}$ as the binary ground-truth label:

**Weighted Binary Cross-Entropy:**

$$
\mathcal{L}_\text{BCE} = -\frac{1}{|\Omega|} \sum_{\mathbf{x} \in \Omega} w(\mathbf{x}) \left[ y(\mathbf{x}) \log p(\mathbf{x}) + \bigl(1 - y(\mathbf{x})\bigr) \log\bigl(1 - p(\mathbf{x})\bigr) \right]
$$

where the weight map is:

$$
w(\mathbf{x}) = w_c(\mathbf{x}) + w_0 \exp\!\left(-\frac{(d_1(\mathbf{x}) + d_2(\mathbf{x}))^2}{2\sigma^2}\right)
$$

$w_c(\mathbf{x})$ is the class-frequency re-balancing weight (inversely proportional to the frequency of the class of pixel $\mathbf{x}$, so rare foreground pixels receive higher weight than common background pixels). $d_1(\mathbf{x})$ and $d_2(\mathbf{x})$ are the distances from pixel $\mathbf{x}$ to the boundaries of the nearest and second-nearest cells, respectively. $w_0 = 10$ sets the magnitude of the boundary-gap penalty, and $\sigma \approx 5$ pixels controls its spatial width. The exponential term gives large weight to the narrow gap between touching cells.

The **Dice loss** directly optimizes the overlap between predicted segmentation mask and ground truth. Using the same $\Omega$ (image pixel set), $p(\mathbf{x})$ (predicted foreground probability), and $y(\mathbf{x})$ (ground-truth binary label) as above:

**Dice Loss:**

$$
\mathcal{L}_\text{Dice} = 1 - \frac{2\sum_{\mathbf{x} \in \Omega} p(\mathbf{x})\, y(\mathbf{x})}{\sum_{\mathbf{x} \in \Omega} p(\mathbf{x}) + \sum_{\mathbf{x} \in \Omega} y(\mathbf{x}) + \epsilon}
$$

$\epsilon \approx 10^{-6}$ is a small constant that prevents division by zero when both the prediction and ground truth are empty. The Dice coefficient equals 1 for perfect overlap and 0 for disjoint predictions. Unlike cross-entropy, Dice is naturally invariant to class imbalance, critical when cells occupy only a small fraction of pixels.

In practice, the two losses are combined:

$$
\mathcal{L} = \lambda_\text{BCE}\,\mathcal{L}_\text{BCE} + \lambda_\text{Dice}\,\mathcal{L}_\text{Dice}
$$

$\lambda_\text{BCE} = \lambda_\text{Dice} = 1$ is the common choice, as cross-entropy provides stable gradients early in training while Dice tightens boundary predictions as training matures.

### DeepCell / Mesmer (2016–2022)

The Van Valen laboratory's DeepCell framework (2016) applied U-Net variants to whole-cell segmentation in fluorescence microscopy, explicitly using two-channel inputs: a nuclear marker (DAPI) and a membrane/cytoplasmic marker. By predicting separate probability maps for nuclei and for whole-cell regions, then combining them, Mesmer (Greenwald et al., 2022) achieved near-human accuracy on tissue segmentation, a long-standing bottleneck for CODEX and multiplexed imaging analysis.

**Limitation** that U-Net-based semantic segmentation could not resolve: *instance separation*. Semantic segmentation assigns a class (cell vs. background) to every pixel, but does not distinguish between touching cells. Two adjacent nuclei produce a connected foreground blob, and the model gives them the same label. This "instance identity" problem required a new generation of solutions.

**Pseudocode, U-Net Forward Pass:**

```python
def unet_forward(x):                          # x: [B, C_in, H, W]
    # --- Encoder ---
    e1 = conv_block(x,  filters=64)           # [B, 64,  H,    W   ]
    e2 = conv_block(pool(e1), filters=128)    # [B, 128, H/2,  W/2 ]
    e3 = conv_block(pool(e2), filters=256)    # [B, 256, H/4,  W/4 ]
    e4 = conv_block(pool(e3), filters=512)    # [B, 512, H/8,  W/8 ]

    # --- Bottleneck ---
    b  = conv_block(pool(e4), filters=1024)   # [B, 1024, H/16, W/16]

    # --- Decoder with skip connections ---
    d4 = conv_block(concat(up(b),  e4), filters=512)
    d3 = conv_block(concat(up(d4), e3), filters=256)
    d2 = conv_block(concat(up(d3), e2), filters=128)
    d1 = conv_block(concat(up(d2), e1), filters=64)

    # --- Output ---
    out = conv_1x1(d1, filters=n_classes)     # [B, n_classes, H, W]
    return sigmoid(out)                        # or softmax for multiclass

# conv_block = Conv2d → BN → ReLU → Conv2d → BN → ReLU
# pool        = MaxPool2d(kernel=2, stride=2)
# up          = ConvTranspose2d(kernel=2, stride=2)  or bilinear upsample
# concat      = torch.cat along channel dimension
```

---

## Era 4, Instance Segmentation (2018 – 2021)

*Semantic segmentation produced connected foreground blobs. Instance segmentation assigns a unique identity to every cell, a much harder problem requiring the model to reason about individual object boundaries simultaneously.*

### The Instance Separation Problem

Instance segmentation requires two things semantic segmentation does not: (1) counting and labeling every distinct object, and (2) producing a separate binary mask for each. General solutions from computer vision, Mask R-CNN (He et al., 2017), run a region-proposal network to generate candidate bounding boxes, then segment each proposal. For cells, this works when cells are large and sparse, but in dense tissue sections with hundreds of overlapping nuclei per field of view, the proposal stage struggles.

Two specialized methods became dominant in cell biology because they encoded cell-specific geometric priors.

### StarDist (Schmidt et al., 2018), MICCAI 2018

StarDist exploits the observation that nuclei are roughly *star-convex*: any line segment from the nucleus center to the boundary lies entirely inside the nucleus. It represents each nucleus by $N$ radial distances from the centroid to the boundary at evenly spaced angles.

A U-Net backbone predicts, for every pixel $\mathbf{x}$:
- **Objectness score** $p(\mathbf{x}) \in [0,1]$: probability that $\mathbf{x}$ is near a nucleus center
- **Radial distance vector** $\hat{\mathbf{r}}(\mathbf{x}) \in \mathbb{R}^N$: predicted distances to the boundary in $N$ directions

Instances are recovered by:
1. Thresholding $p(\mathbf{x})$ to obtain candidate center pixels
2. Reconstructing the star-convex polygon for each candidate from $\hat{\mathbf{r}}$
3. Running **Non-Maximum Suppression (NMS)**: for all pairs of overlapping polygons, discard the one with lower objectness score; repeat until no overlapping pairs remain

**StarDist Loss:**

$$
\mathcal{L}_\text{StarDist} = \underbrace{\mathcal{L}_\text{CE}\!\left(\hat{p}, p_\text{gt}\right)}_{\text{objectness BCE}} + \lambda \underbrace{\sum_{k=1}^{N} \left|\hat{r}_k(\mathbf{x}) - r_k^\text{gt}(\mathbf{x})\right|}_{\text{L1 (MAE) radial distance loss, summed over directions}}
$$

The radial distance regression uses Mean Absolute Error (L1 loss), not the Huber/SmoothL1 loss. The L1 loss is robust to the occasional large-error pixel at cell boundaries without the need for a quadratic regime. The shape loss applies only to foreground (nucleus interior) pixels. $\lambda$ is a balancing coefficient (typically 1.0).

**Limitation** of StarDist: the star-convex assumption breaks for elongated, C-shaped, or highly irregular cells, common in mesenchymal cells, neurons, or immune cells extending filopodia.

### Cellpose (Stringer et al., 2021), Nature Methods 2021

Cellpose dispensed with shape priors entirely, enabling segmentation of arbitrarily shaped cells. Its key innovation is a **gradient-flow representation**: instead of predicting object boundaries directly, the network predicts two scalar fields, the x-component $\hat{v}_x(\mathbf{x})$ and y-component $\hat{v}_y(\mathbf{x})$ of a vector that, at every foreground pixel, points toward the interior of the cell it belongs to.

To recover instances, Cellpose simulates particle dynamics: starting from every pixel in the binary foreground mask, follow the predicted gradient field. All particles that converge to the same attractor basin are assigned to the same cell instance. This reduces instance segmentation to a flow simulation, no NMS, no shape constraints.

The ground-truth flow field is **not** a straight-line vector to the centroid. A naive Euclidean unit vector $(\mathbf{c}-\mathbf{x})/\|\mathbf{c}-\mathbf{x}\|$ can point completely outside the cell body for non-convex or irregular shapes, causing the particle simulation to escape the cell mask. Instead, Cellpose generates targets by simulating **heat diffusion** originating from the cell's median pixel $\mathbf{x}_0$:

**Target Vector Field via Heat Diffusion:**

**Step 1, Initialize:** $T^{(0)}(\mathbf{x}) = \mathbf{1}[\mathbf{x} = \mathbf{x}_0]$ (unit temperature at the cell's median pixel, zero elsewhere in the mask $\Omega_i$).

$$
T^{(d+1)}(\mathbf{x}) = \frac{1}{|\mathcal{N}(\mathbf{x})|} \sum_{\mathbf{e}\,\in\,\{\pm\mathbf{e}_x,\,\pm\mathbf{e}_y\}} T^{(d)}(\mathbf{x}+\mathbf{e})\;\mathbf{1}[\mathbf{x}+\mathbf{e}\in\Omega_i]
$$

**Step 2, Diffuse** for $D$ steps (absorbing boundary: pixels outside $\Omega_i$ remain at 0), then:

$$
\mathbf{v}^\text{gt}(\mathbf{x}) = \frac{\nabla T^{(D)}(\mathbf{x})}{\|\nabla T^{(D)}(\mathbf{x})\|_2}
\quad \text{for every foreground pixel } \mathbf{x} \in \Omega_i
$$

$\mathcal{N}(\mathbf{x})$ is the set of in-mask neighbors of pixel $\mathbf{x}$; $\mathbf{e}_x, \mathbf{e}_y$ are unit steps along each axis. Because heat diffuses only within the cell mask $\Omega_i$, the resulting gradient $\nabla T^{(D)}$ curves smoothly around cell boundaries and never exits the mask, even for concave or highly irregular cell shapes. This property is precisely what makes gradient-flow particle simulation robust.

**Cellpose Training Loss:**

$$
\mathcal{L}_\text{Cellpose} = \underbrace{\left\| \hat{\mathbf{v}}(\mathbf{x}) - \mathbf{v}^\text{gt}(\mathbf{x}) \right\|_2^2}_{\text{flow MSE (foreground pixels only)}} + \underbrace{\mathcal{L}_\text{BCE}\!\left(\hat{p}(\mathbf{x}), p_\text{gt}(\mathbf{x})\right)}_{\text{cell probability BCE}}
$$

The flow loss is computed only over ground-truth foreground pixels. The probability loss trains the companion head that generates the binary foreground mask used to seed the flow simulation at inference time.

**Pseudocode, Cellpose Inference (Gradient Flow):**

```python
def cellpose_inference(image):
    # 1. Forward pass: predict vector field + cell probability
    v_x, v_y, p = model(image)            # each [H, W]

    # 2. Get binary foreground mask
    mask = sigmoid(p) > 0.5               # [H, W] bool

    # 3. Simulate gradient flow for T steps
    T = 200
    coords = get_foreground_coords(mask)  # [N_px, 2]
    for t in range(T):
        # interpolate gradient at current positions
        grad = interpolate_field(v_x, v_y, coords)   # [N_px, 2]
        coords = coords + grad * step_size           # follow flow

    # 4. Cluster converged coordinates into instances
    # pixels whose final coordinates are within r=1 px of each other
    # belong to the same cell
    labels = connected_components(coords, radius=1.0)  # [N_px] int
    return fill_label_map(labels, mask, shape=image.shape)
```

Cellpose's generality, it segments any cell type, in any staining, in 2D or 3D, made it the de facto standard for fluorescence microscopy cell segmentation by 2022. The **remaining limitation**: Cellpose, like all supervised methods, requires annotated training data that matches the target domain. Cellpose 2.0 (Pachitariu & Stringer, 2022) addressed this with a human-in-the-loop retraining workflow, but obtaining ground-truth labels for dense tissue sections, 3D volumes, or novel marker combinations remained expensive.

### Image Restoration: CARE (2018)

Content-Aware Image Restoration (CARE; Weigert et al., 2018) showed that a U-Net could learn to restore high-quality fluorescence images from paired low/high-quality acquisitions, effectively denoising, deconvolving, and upsampling in one learned operation. This matters enormously in practice: better input images produce better segmentations downstream without any modification to the segmentation model.

CARE was trained on matched pairs $(\tilde{I}, I^*)$ where $\tilde{I}$ is a noisy/degraded image and $I^*$ is the high-quality target, minimizing mean squared error:

**CARE Restoration Loss:**

$$
\mathcal{L}_\text{CARE} = \frac{1}{|\Omega|} \sum_{\mathbf{x} \in \Omega} \left\| f_\theta(\tilde{I})(\mathbf{x}) - I^*(\mathbf{x}) \right\|_2^2
$$

$\Omega$ is the set of all pixels in the image and $|\Omega|$ is the total pixel count (same convention as in the U-Net losses above). $f_\theta$ is the U-Net with learned parameters $\theta$; $f_\theta(\tilde{I})(\mathbf{x})$ denotes its predicted intensity at pixel $\mathbf{x}$ given the degraded input $\tilde{I}$. $I^*(\mathbf{x})$ is the corresponding ground-truth intensity from the high-quality acquisition. The key limitation: acquiring paired low/high-quality data is expensive. Self-supervised methods (Noise2Void, 2019; Noise2Self, 2019) later removed this requirement by exploiting the statistical independence of noise at different pixels to train without any clean reference images.

---

## Era 5, Vision Transformers (2020 – 2022)

*Transformers from NLP crossed into vision. Global self-attention replaced local convolutions, capturing long-range dependencies across the entire image field, particularly valuable for spatial context in tissue analysis.*

### The Limitation Transformers Resolve

Convolutional layers are inherently *local*: a $3\times3$ convolution aggregates information from a $3\times3$ neighborhood. To cover the full image, many layers must be stacked, each expanding the effective receptive field only incrementally. In practice, CNNs struggle to exploit spatial relationships between cells that are far apart, for instance, correlating the phenotype of an immune cell with a tumor cell 500 microns away. Transformers, by contrast, compute *global self-attention* across all positions simultaneously.

### Vision Transformer (ViT, Dosovitskiy et al., 2021), ICLR 2021

ViT applies a standard Transformer encoder to images by first splitting the image into a regular grid of non-overlapping patches of size $P \times P$ pixels. Each patch is flattened and linearly projected to a $d$-dimensional embedding vector. The resulting sequence of $N = (H \cdot W) / P^2$ patch tokens, augmented with a learnable class token [CLS] and positional encodings, is processed by $L$ Transformer blocks.

**ViT Patch Sequence:**

$$
\mathbf{T}_0 = \left[ \mathbf{e}_\text{cls};\;\; \mathbf{E}\,\mathbf{x}_1^\text{patch};\;\; \mathbf{E}\,\mathbf{x}_2^\text{patch};\;\; \ldots;\;\; \mathbf{E}\,\mathbf{x}_N^\text{patch} \right] + \mathbf{E}_\text{pos}
$$

$\mathbf{T}_0 \in \mathbb{R}^{(N+1) \times d}$ is the initial token matrix fed into the Transformer. $\mathbf{x}_i^\text{patch} \in \mathbb{R}^{P^2 C}$ is the flattened $i$-th patch ($C$ input channels). $\mathbf{E} \in \mathbb{R}^{d \times P^2 C}$ is the learned linear projection. $\mathbf{E}_\text{pos} \in \mathbb{R}^{(N+1) \times d}$ are learnable positional encodings. The [CLS] token $\mathbf{e}_\text{cls}$ aggregates global information. (We use $\mathbf{T}$ for the token sequence to distinguish it from latent vectors $\mathbf{z}$ used in generative models later.)

Each Transformer block applies **Multi-Head Self-Attention (MHSA)** followed by a feed-forward MLP. In self-attention, each token queries all other tokens:

**Scaled Dot-Product Self-Attention:**

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}
$$

$\mathbf{Z} \in \mathbb{R}^{(N+1) \times d}$ is the token matrix entering the current Transformer block (i.e., the output of the previous block, or $\mathbf{T}_0$ at the first block). Queries $\mathbf{Q} = \mathbf{Z}\mathbf{W}_Q$, keys $\mathbf{K} = \mathbf{Z}\mathbf{W}_K$, and values $\mathbf{V} = \mathbf{Z}\mathbf{W}_V$ are linear projections of $\mathbf{Z}$ via learned weight matrices $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d \times d_k}$. $d_k = d / n_\text{heads}$ is the per-head projection dimension. The softmax is applied row-wise over the $(N+1)$ key positions, so each of the $N+1$ query tokens attends to all $N+1$ positions simultaneously. In multi-head attention, $n_\text{heads}$ parallel attention functions are computed independently and their outputs concatenated along the feature dimension.

**ViT's limitation** for microscopy: (1) quadratic computational complexity $\mathcal{O}(N^2 d)$ in sequence length, a 256×256 image with 16×16 patches gives $N=256$ tokens, manageable; but a 2048×2048 tissue crop gives $N=16384$, which is computationally prohibitive; (2) ViT lacks the inductive biases (locality, translation equivariance) that make CNNs data-efficient, it needs either massive datasets or sophisticated self-supervised pretraining to perform well.

### Swin Transformer (Liu et al., 2021), ICCV 2021

The Swin Transformer resolved ViT's complexity and scale problems with two innovations: **local windowed attention** and a **hierarchical feature map**.

Instead of attending globally, Swin partitions the token sequence into non-overlapping windows of size $M \times M$ and applies self-attention only within each window:

**Swin vs. ViT Complexity:**

$$
\text{Swin: } \mathcal{O}\!\left(4HWd^2 + 2M^2 HW d\right) \quad \text{vs.} \quad \text{ViT: } \mathcal{O}\!\left(4HWd^2 + 2(HW)^2 d\right)
$$

For a feature map of $H \times W$ tokens with hidden dimension $d$ and window size $M$. Since $M \ll HW$, Swin scales linearly with image size; ViT scales quadratically. This makes Swin practical for gigapixel pathology images.

To allow cross-window information flow, alternate Transformer blocks use a **shifted window**: windows are displaced by $(\lfloor M/2 \rfloor, \lfloor M/2 \rfloor)$ pixels relative to the previous layer, so tokens near window boundaries in one layer become window-interior tokens in the next. A cyclic-shift trick with masking makes this computationally efficient.

Additionally, Swin merges adjacent $2\times2$ patches after each stage (like strided convolution), generating a hierarchy of feature maps at resolutions $\frac{H}{4}\times\frac{W}{4}$, $\frac{H}{8}\times\frac{W}{8}$, $\frac{H}{16}\times\frac{W}{16}$, $\frac{H}{32}\times\frac{W}{32}$, compatible with FPN-based detection and U-Net-like decoder heads for dense prediction.

**Pseudocode, Swin Window Self-Attention:**

```python
def swin_block(x, shift=False):
    # x: [B, H, W, d]  feature map as spatial grid

    # 1. Optional cyclic shift
    if shift:
        x = roll(x, shift=(-M//2, -M//2))  # shift tokens for cross-window mixing

    # 2. Partition into M×M windows
    windows = partition(x, M)              # [B * nW, M*M, d]  (nW = H*W/M^2)

    # 3. Self-attention within each window
    attn_out = self_attention(windows)     # [B * nW, M*M, d]

    # 4. Reverse window partition
    x = reverse_partition(attn_out, H, W) # [B, H, W, d]

    # 5. Unmask cyclic shift
    if shift:
        x = roll(x, shift=(M//2, M//2))

    # 6. Feed-forward MLP
    x = x + mlp(layer_norm(x))
    return x

# Alternating: swin_block(shift=False) → swin_block(shift=True) → ...
```

Swin rapidly became the backbone of choice for histopathology slide analysis (e.g., TransPath, HierarchicalImagePyramid) because its hierarchical multi-scale features map naturally to the multiple magnification levels of whole-slide images.

---

## Era 6, Self-Supervised Learning & Foundation Models (2022 – 2024)

*Pretraining on millions of unlabeled images with self-supervised objectives produced universal microscopy representations. Foundation models trained on 100,000+ slides enabled zero-shot and few-shot deployment across tasks and institutions.*

### The Annotation Bottleneck

Supervised training for microscopy is inherently expensive. Expert annotators take minutes per image for cell segmentation and hours per slide for histological grade labeling. Rare pathologies yield only tens of annotated examples. Self-supervised learning pre-trains representations from unlabeled images, dramatically reducing the labeled data needed for downstream tasks.

### Masked Autoencoders (MAE, He et al., 2022), CVPR 2022

MAE applies ViT to a simple reconstruction objective: randomly mask 75% of image patches, pass only visible patches through a ViT encoder, then decode the masked patches from the latent representations using a lightweight decoder. The model learns to predict the missing pixels from context.

**MAE Reconstruction Loss:**

$$
\mathcal{L}_\text{MAE} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \left\| f_\theta(\mathbf{x}_{\bar{\mathcal{M}}})_i - \mathbf{x}_i \right\|_2^2
$$

$\mathcal{M}$ is the set of masked patch indices; $\bar{\mathcal{M}}$ is the set of visible patch indices. The loss is computed only on masked patches (this makes the task harder than computing it everywhere, forcing the encoder to build richer representations). Pixel values are normalized per-patch before computing the loss.

For microscopy, MAE-pretrained ViTs initialize downstream segmentation models and need only a fraction of the labeled data that a randomly initialized network requires to reach the same performance.

### DINO / DINOv2: Self-Distillation Without Labels

DINO (Caron et al., 2021) introduced *self-distillation*: a student ViT is trained to match the output distribution of a momentum-averaged teacher ViT on different random augmentations of the same image. The teacher is never gradient-updated, its weights are an exponential moving average (EMA) of the student, $\theta_t \leftarrow m\,\theta_t + (1-m)\,\theta_s$, with momentum $m \approx 0.996$. No labels are ever used. The resulting features exhibit striking emergent properties: DINO attention maps automatically delineate foreground objects because the cross-view matching objective forces the encoder to identify semantic regions that remain consistent across augmentations.

The **limitation** of the original DINO: it was validated on curated datasets (ImageNet), and naïvely scaling to uncurated web crawls degraded feature quality substantially. DINOv2 (Oquab et al., 2023) resolves this with two contributions: (1) a principled *data curation pipeline* and (2) a richer three-term training objective.

#### DINOv2: Data Curation (LVD-142M)

Starting from a large uncurated web-crawled image corpus, DINOv2 curates a 142M-image dataset (LVD-142M) by using a self-supervised model pretrained on ImageNet-22k as a retrieval backbone: for each curated seed image, the nearest neighbors in the uncurated pool are retrieved by cosine similarity in the embedding space. Retrieved images pass a deduplication and quality filter. The key insight is that *image retrieval with a self-supervised backbone* selects semantically coherent content without any human annotation, allowing DINOv2 to close the gap with weakly supervised models (CLIP, OpenCLIP) that rely on billions of noisy text-image pairs.

#### DINOv2: Three-Term Training Objective

DINOv2 trains a ViT student against an EMA teacher with a combined loss over global and local image crops:

**Image-Level CLS Distillation (DINO term):**

$$
\mathcal{L}_\text{DINO} = -\sum_{k=1}^{K} p_{t,k}\,\log p_{s,k}
$$

The cross-entropy is averaged over all valid student–teacher crop pairs (teacher uses only global crops; student uses both global and local crops). $K$ is the number of prototype dimensions. $p_{t,k}$ is the teacher's softmax probability over prototype $k$ for the [CLS] token of its crop view, computed with a sharpened temperature $\tau_t$ and Sinkhorn–Knopp centering to prevent collapse. $p_{s,k}$ is the student's corresponding distribution (no centering). This loss aligns the global semantic content of different augmented views of the same image.

**Patch-Level Masked Prediction (iBOT term):**

$$
\mathcal{L}_\text{iBOT} = -\frac{1}{|\mathcal{M}|}\sum_{i\,\in\,\mathcal{M}}\sum_{k=1}^{K} p_{t,i,k}\,\log p_{s,i,k}
$$

$\mathcal{M}$ is the set of randomly masked patch positions in the student's input (analogous to MAE masking, but the targets here are teacher prototype distributions rather than raw pixels). The teacher receives the full unmasked image and outputs patch-level prototype distributions $p_{t,i,k}$ over $K$ prototypes at each patch position $i$. The student processes the masked image and must predict those distributions at the masked positions, yielding $p_{s,i,k}$. This term adds a local, dense self-supervised signal complementary to the global CLS loss.

**KoLeo Regularizer (feature-space spreading):**

$$
\mathcal{L}_\text{KoLeo} = -\frac{1}{n}\sum_{i=1}^{n}\log d_{n,i}, \qquad d_{n,i} = \min_{j \neq i}\|\mathbf{x}_i - \mathbf{x}_j\|_2
$$

$n$ is the number of samples in the mini-batch. $\mathbf{x}_i$ is the $\ell_2$-normalized [CLS] feature vector of sample $i$. $d_{n,i}$ is the minimum L2 distance from $\mathbf{x}_i$ to any other sample in the batch. Maximizing $\log d_{n,i}$ encourages features to spread uniformly across the unit hypersphere, counteracting representation collapse and cluster degeneration without requiring explicit contrastive pairs.

**DINOv2 Combined Loss:**

$$
\mathcal{L}_\text{DINOv2} = \mathcal{L}_\text{DINO} + \mathcal{L}_\text{iBOT} + \lambda_\text{KoLeo}\,\mathcal{L}_\text{KoLeo}
$$

$\lambda_\text{KoLeo}$ controls the strength of the feature-spreading regularizer. The DINO term captures global semantics, the iBOT term captures local dense structure, and the KoLeo term prevents feature collapse, together producing features that excel at both classification and dense prediction simultaneously.

DINOv2 scales to a ViT-g/14 backbone (1.1B parameters, 1536-dimensional embeddings, 24 attention heads, 14×14 pixel patch size). On ImageNet linear probing with frozen features, ViT-g/14 achieves 86.5% top-1 accuracy, matching weakly supervised OpenCLIP ViT-G (86.2%) without any text supervision. On dense prediction (ADE20k semantic segmentation, linear probe), DINOv2 reaches 53.0 mIoU vs. 47.5 for prior SSL methods. Domain generalization improves by +29.6% on ImageNet-A and +22.1% on ImageNet-R over previous self-supervised approaches.

In computational pathology, DINOv2 became the *de facto* training recipe: UNI, CONCH, MOOZY's tile encoder, and GigaPath's tile encoder all adopt the DINOv2 objective on domain-specific corpora. The combination of curated data at scale, CLS+patch self-distillation, and KoLeo regularization produces the most transferable frozen features currently available for H&E and related imaging tasks.

### Segment Anything Model (SAM, Kirillov et al., 2023), ICCV 2023

SAM was trained on 1 billion masks from 11 million natural images, resulting in a model that produces high-quality segmentation masks from three types of prompts: (1) point clicks, (2) bounding boxes, or (3) a prior rough mask. It cannot be expected to handle the contrast and morphology of fluorescence microscopy out of the box, but it was immediately adapted (Cellpose + SAM, MedSAM, CellSAM) to cell images with minimal fine-tuning on domain-specific data. SAM 2 (2024) extended the architecture to video, enabling time-lapse cell tracking by propagating masks across frames.

### Pathology Foundation Models

Large-scale pretraining on pathology-specific data produced a new generation of foundation models that dominate almost every H&E task benchmark:

**UNI** *(Nature Medicine 2024)*
Trained via DINOv2 on 100,000+ H&E slides (450 million tiles). A ViT-L encoder. Evaluated on 34 computational pathology tasks spanning 20+ cancer types, outperforming task-specific models across the board. Demonstrates that scale and self-supervision together eliminate the need for task-specific pretraining.

**CONCH** *(Nature Medicine 2024)*
A vision-language model. Pretrained with contrastive learning on over one million image-caption pairs drawn from pathology reports and open-access literature. Enables zero-shot classification and retrieval: given a text query "ductal carcinoma in situ," CONCH retrieves matching slide regions without any labeled examples.

**PLIP** *(Nature Medicine 2023)*
Pathology Language-Image Pretraining. Trained on 208,414 image-text pairs from social media (Twitter/X pathology posts). Shows that even noisy web-scraped data enables useful pathology representations, lower cost than curating clinical reports.

**Prov-GigaPath** *(Nature 2024)*
A large-scale ViT model trained on 1.3 million whole-slide images from Providence Health network, the largest pathology training set to date. Uses a two-stage architecture: a tile encoder (ViT-g pretrained with DINOv2) followed by a slide-level aggregation encoder that processes the long sequence of tile embeddings with flash-attention for efficiency.

**MOOZY** *(arXiv 2026)*
A patient-first foundation model (Kotp et al., 2026). MIL architectures such as ABMIL and TransMIL have long aggregated multi-slide patient data for specific downstream tasks; MOOZY's novelty is shifting this aggregation into the *pretraining* phase itself. A lightweight case transformer pools per-slide embeddings from a frozen ViT-S tile encoder (DINOv2) into a patient-level representation, pretrained via masked self-distillation on 77K public slides and supervised across 333 classification and survival tasks from 56 datasets. Achieves +7.4% weighted F1 over TITAN at 14× fewer parameters than GigaPath (85.8M vs. 1.22B), though the benefit of cross-slide aggregation remains to be quantified on tasks specifically designed for multi-slide reasoning.

**Remaining limitation** shared across all these models: they were trained predominantly on H&E images and do not transfer well to fluorescence microscopy or multiplexed imaging without domain adaptation. Foundation models for fluorescence and spatial proteomics remain an active frontier.


## H&E Histopathology: From Whole-Slide Images to Clinical Prediction

*Gigapixel whole-slide images cannot fit into GPU memory. Multiple instance learning circumvents this by treating the slide as a bag of tiles and aggregating tile-level features into slide-level predictions.*

### The Whole-Slide Image Problem

A 40× whole-slide image (WSI) at 0.25 μm/pixel for a 2 cm × 2 cm tissue section is roughly 80,000 × 80,000 pixels, around 6.4 gigapixels. Standard GPU memory holds perhaps 256 × 256 pixels at 40×. Direct end-to-end training on WSIs is impossible. Two complementary strategies dominate:

1. **Patch-based pipelines:** Extract non-overlapping 256×256 or 512×512 tiles, embed each with a pretrained CNN or ViT encoder, then aggregate the bag of tile embeddings into a slide-level prediction
2. **Hierarchical / pyramid models:** Use multiple magnifications simultaneously, leveraging coarse-scale context to weight fine-scale features

### Multiple Instance Learning (MIL)

In MIL, a slide-level label (cancer grade, mutation status, survival) is weakly associated with all tiles, but only a subset of tiles carry the relevant signal. The model must identify and aggregate these *key instances*. CLAM (Lu et al., 2021) introduced attention-based MIL for pathology:

**Attention MIL (CLAM):**

$$
a_k = \frac{\exp\!\left(\mathbf{w}^\top \tanh\!\left(\mathbf{V}\,\mathbf{h}_k\right) \odot \text{sigm}\!\left(\mathbf{U}\,\mathbf{h}_k\right)\right)}{\sum_{j=1}^{K} \exp\!\left(\mathbf{w}^\top \tanh\!\left(\mathbf{V}\,\mathbf{h}_j\right) \odot \text{sigm}\!\left(\mathbf{U}\,\mathbf{h}_j\right)\right)}
$$

$\mathbf{h}_k \in \mathbb{R}^d$ is the embedding of tile $k$ (from a pretrained ResNet or ViT); $\mathbf{V}, \mathbf{U} \in \mathbb{R}^{L \times d}$ and $\mathbf{w} \in \mathbb{R}^L$ are learnable; $\odot$ is element-wise product; $\text{sigm}(\cdot)$ is sigmoid. The gated attention (product of $\tanh$ and sigmoid branches) is more expressive than single-branch attention. The slide-level representation is the weighted sum: $\mathbf{z} = \sum_k a_k \mathbf{h}_k$.

The key advance of CLAM beyond naive max-pooling MIL: the attention weights $a_k$ are interpretable, they highlight which tissue regions drove the prediction, providing pathologists with a spatially explicit explanation.

### Task Diversity in H&E AI

**Cancer Detection**
Binary classification: tumor present/absent. Training data: pathologist-annotated slides. Current best models approach pathologist-level sensitivity (>95%) for common cancers (prostate, breast, colorectal). Key challenge: high specificity to avoid false positives that trigger unnecessary procedures.

**Grading & Subtyping**
Gleason grading (prostate), ISUP grade, breast receptor subtype. Models such as Google's PANDA challenge winner (2020) achieve pathologist-concordant Gleason grade at 98% agreement. Foundation models (UNI, CONCH) further improve few-shot grading accuracy by 10–20%.

**Biomarker Prediction**
Predicting genomic alterations (MSI-H, BRCA1/2, TMB, copy number profiles) directly from H&E morphology, avoiding expensive sequencing. A 2019 Nature Genetics study showed pan-cancer mutation prediction from slide images; foundation models extended this to 200+ genomic features across TCGA.

**Survival Prediction**
Predicting patient prognosis from WSI alone. Methods combine attention MIL with Cox proportional hazard losses or rank-based losses. SurvPath (2023) integrates WSI features with transcriptomic profiles for multi-modal survival prediction.

**Segmentation in Pathology**
Delineating tumor regions, gland structures, stroma, necrosis, and invasive margin. Panoptic segmentation (semantic + instance) is increasingly used. Hover-Net (2019) predicts nuclear types and instances simultaneously via dual U-Net branches for classification and geometry.

**Spatial Context**
Mapping the tumor microenvironment: density and proximity of immune cells, fibroblasts, and tumor cells. Graph-based models capture spatial relationships between cells across the slide for immune-hot/cold classification predicting immunotherapy response.

---

## Open Challenges

**Domain Shift**
Models trained on one scanner, staining protocol, or tissue type fail on others. Stain normalization methods (Macenko, Vahadane) help for H&E; for fluorescence, there is no equivalent standard. Domain-adaptive training, test-time adaptation, and foundation model fine-tuning are active areas.

**3D Segmentation**
Most methods are 2D. Tissue sections are inherently 3D, and confocal z-stacks or light-sheet volumes require true 3D segmentation. 3D Cellpose and 3D StarDist exist but are computationally expensive and require 3D annotated training data, which is even harder to generate than 2D ground truth.

**Annotation Cost**
Expert annotations remain the bottleneck. Active learning, few-shot learning, and self-supervised pretraining are all strategies to reduce this dependency. SAM-based interactive annotation tools partially address it, but true zero-shot generalization to novel cell types and stains remains unsolved.

**Interpretability**
A model that classifies a tumor as grade-3 must explain which morphological features drove that decision to be clinically deployable. Attention maps (CLAM) and concept bottleneck models partially address this, but deep feature explanations remain post-hoc and qualitative, not causal.

**Multi-Modal Foundation Models**
No foundation model yet spans H&E, fluorescence, CODEX, and electron microscopy simultaneously. Each modality has distinct image formation physics requiring separate pretraining. Cross-modal foundation models that learn unified representations across all microscopy types are an emerging frontier.

**Prospective Validation**
Most published AI models are evaluated on retrospective cohorts with known outcomes. Regulatory approval requires prospective clinical trials. The gap between benchmark accuracy and clinical deployment remains wide, partly a data issue, partly a model calibration issue (overconfident predictions).

---

## Summary: Major Advances and Remaining Challenges

Two decades of computational microscopy have moved the field from brittle, hand-tuned pipelines to general-purpose models that learn directly from data. A handful of advances stand out.

**Major advances**

- **From rules to representations.** Watershed and CellProfiler gave way to learned features, first via U-Net's encoder-decoder with skip connections, which made biomedical segmentation tractable from hundreds (not thousands) of labeled examples.
- **Instance segmentation that handles real cell shapes.** StarDist (star-convex polygons), Cellpose (gradient flows), and Mesmer (nuclear + membrane fusion) closed the gap between semantic masks and per-cell instances, including for densely packed and irregular cells.
- **Transformers and global context.** Vision Transformers and Swin made it practical to model long-range dependencies in gigapixel images, and attention-MIL methods like CLAM turned whole-slide images into slide-level predictions with interpretable evidence.
- **Self-supervision and foundation models.** MAE, DINO/DINOv2, and SAM showed that pretraining on unlabeled images yields features that transfer broadly; UNI, CONCH, and Prov-GigaPath ported this recipe to histopathology, and MOOZY pushed aggregation up to the patient level.
- **Spatial proteomics goes quantitative.** CODEX-style multiplexed imaging coupled to deep segmentation and normalization pipelines now produces single-cell, multi-marker maps inside intact tissue, bringing morphology and protein expression into the same coordinate frame.

**Remaining challenges**

- **Domain shift and generalization.** Models still degrade across scanners, stains, and tissues; robust normalization for fluorescence and reliable test-time adaptation remain open.
- **3D at scale.** Confocal stacks, light-sheet volumes, and volume EM demand true 3D models, but annotated 3D ground truth is scarce and 3D inference is computationally heavy.
- **Annotation bottleneck.** Even with SAM and self-supervision, expert labels are the rate-limiting resource for new cell types, rare phenotypes, and novel modalities.
- **Interpretability and trust.** Attention maps and concept bottlenecks help, but causal, clinically actionable explanations of deep features are not yet standard.
- **Unified multi-modal foundations.** No single model spans H&E, fluorescence, CODEX, 10x-Visium, and electron microscopy; image-formation physics differ enough that cross-modality pretraining is still an open problem.
- **Prospective clinical validation.** Most reported performance comes from retrospective cohorts. Prospective trials, careful calibration, and regulatory pathways are what separate benchmark wins from deployed tools.

> **Key Insight:** The field's trajectory is from narrow, modality-specific tools toward general representations that can be steered with little supervision. The next decade will be judged less by new architectures than by whether these models survive deployment, across labs, scanners, modalities, and patients.

---

## Summary: Methods Across the Field

| Method | Year | Architecture | Task | Semantic Seg | Instance Seg | Slide-level | 3D | Key Innovation |
|--------|------|-------------|------|:---:|:---:|:---:|:---:|----------------|
| Watershed + CellProfiler | 2006 | Classical CV | Cell detection | ~ | ~ | ✗ | ✗ | Modular handcrafted pipeline |
| AlexNet / VGG (applied) | 2012–14 | CNN | Patch classification | ✗ | ✗ | ✗ | ✗ | Learned features beat handcrafted |
| U-Net | 2015 | Encoder-decoder CNN | Biomedical seg | ✓ | ✗ | ✗ | ~ | Skip connections, data augmentation |
| CARE | 2018 | U-Net | Image restoration | ✗ | ✗ | ✗ | ✓ | Content-aware denoising/deconvolution |
| StarDist | 2018 | U-Net + polygon head | Nucleus instance seg | ✓ | ✓ | ✗ | ✓ | Star-convex polygon representation + NMS |
| Mesmer (DeepCell) | 2022 | U-Net | Whole-cell seg | ✓ | ✓ | ✗ | ✗ | Two-channel nuclear + membrane input |
| Cellpose | 2021 | U-Net | Universal cell seg | ✓ | ✓ | ✗ | ✓ | Gradient flow representation |
| CLAM | 2021 | ResNet + Attention MIL | Slide classification | ✗ | ✗ | ✓ | ✗ | Gated attention MIL + interpretability |
| ViT | 2021 | Transformer | Classification | ~ | ✗ | ~ | ✗ | Global self-attention over image patches |
| Swin Transformer | 2021 | Hierarchical Transformer | Dense prediction | ✓ | ✓ | ✓ | ✗ | Shifted windows, linear complexity |
| MAE / DINO / DINOv2 | 2021–23 | ViT (SSL) | Pretraining | ~ | ~ | ✓ | ✗ | MAE: pixel reconstruction; DINOv2: CLS distillation + iBOT patch MIM + KoLeo regularizer; curated LVD-142M |
| SAM | 2023 | ViT + prompt encoder | Promptable seg | ✓ | ✓ | ✗ | ✗ | 1B mask training; zero-shot adaptability |
| UNI / CONCH | 2024 | ViT-L / ViT + language | Pathology foundation | ~ | ✗ | ✓ | ✗ | 100K+ slide pretraining; zero-shot transfer |

*✓ = native support; ~ = partial / with adapter; ✗ = not supported natively*

---

## References

1. **Carpenter A.E. et al.** (2006). CellProfiler: image analysis software for identifying and quantifying cell phenotypes. *Genome Biology*, 7, R100.

2. **Long J., Shelhamer E., Darrell T.** (2015). Fully convolutional networks for semantic segmentation. *CVPR 2015*.

3. **Ronneberger O., Fischer P., Brox T.** (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI 2015*.

4. **Goltsev Y. et al.** (2018). Deep profiling of mouse splenic architecture with CODEX multiplexed imaging. *Cell*, 174(4), 968–981.

5. **Schmidt U. et al.** (2018). Cell detection with star-convex polygons. *MICCAI 2018 Workshops*. (Full paper: Weigert M. et al., 2022, *Nature Methods*.)

6. **Weigert M. et al.** (2018). Content-aware image restoration: pushing the limits of fluorescence microscopy. *Nature Methods*, 15, 1090–1097.

7. **Dosovitskiy A. et al.** (2021). An image is worth 16×16 words: Transformers for image recognition at scale. *ICLR 2021*.

8. **Stringer C. et al.** (2021). Cellpose: a generalist algorithm for cellular segmentation. *Nature Methods*, 18, 100–106.

9. **Liu Z. et al.** (2021). Swin Transformer: Hierarchical vision transformer using shifted windows. *ICCV 2021*.

10. **Caron M. et al.** (2021). Emerging properties in self-supervised vision transformers. *ICCV 2021*. (DINO)

11. **Lu M.Y. et al.** (2021). Data-efficient and weakly supervised computational pathology on whole-slide images. *Nature Biomedical Engineering*, 5, 555–570. (CLAM)

12. **Greenwald N.F. et al.** (2022). Whole-cell segmentation of tissue images with human-level performance using large-scale data annotation and deep learning. *Nature Biotechnology*, 40, 555–565. (Mesmer)

13. **He K. et al.** (2022). Masked autoencoders are scalable vision learners. *CVPR 2022*.

14. **Pachitariu M., Stringer C.** (2022). Cellpose 2.0: how to train your own model. *Nature Methods*, 19, 1634–1641.

15. **Kirillov A. et al.** (2023). Segment Anything. *ICCV 2023*.

16. **Huang Z. et al.** (2023). A visual-language foundation model for pathology image analysis using medical Twitter. *Nature Medicine*, 29, 2307–2316. (PLIP)

17. **Chen R.J. et al.** (2024). Towards a general-purpose foundation model for computational pathology. *Nature Medicine*, 30, 850–862. (UNI)

18. **Lu M.Y. et al.** (2024). A visual-language foundation model for computational pathology. *Nature Medicine*, 30, 863–874. (CONCH)

19. **Xu H. et al.** (2024). A whole-slide foundation model for digital pathology from real-world data. *Nature*, 630, 181–188. (Prov-GigaPath)

20. **Oquab M. et al.** (2023). DINOv2: Learning robust visual features without supervision. *Transactions on Machine Learning Research (TMLR)*, 2024. arXiv:2304.07193.

21. **Kotp Y. et al.** (2026). MOOZY: A patient-first foundation model for computational pathology. *arXiv:2603.27048*.

