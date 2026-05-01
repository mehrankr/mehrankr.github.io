---
author: Mehran Karimzadeh
date: April 30, 2026
description: A historical review of explainable AI methods, from gradient saliency maps to sparse autoencoders and circuit tracing in frontier models.
---

# Interpretability of Deep Learning Models: A Historical Review

**From Gradient Saliency to Sparse Autoencoders (2013–2026)**

---

## Contents

1. [Why Does Interpretability Matter?](#why-does-interpretability-matter)
2. [A Taxonomy of Methods](#a-taxonomy-of-methods)
3. [Historical Timeline](#historical-timeline)
4. **Era 1 — Gradient-Based Methods (2013–2016)**
   - [Saliency Maps](#saliency-maps)
   - [Gradient × Input](#gradient--input)
   - [Guided Backpropagation](#guided-backpropagation)
   - [Layer-wise Relevance Propagation (LRP)](#layer-wise-relevance-propagation-lrp)
5. **Era 2 — Attribution & Model-Agnostic Methods (2016–2017)**
   - [LIME](#lime)
   - [Grad-CAM](#grad-cam)
   - [DeepLIFT](#deeplift)
   - [Integrated Gradients](#integrated-gradients)
   - [SHAP, GradSHAP & DeepSHAP](#shap-gradshap--deepshap)
6. **Era 3 — Concept-Based Methods (2016–2020)**
   - [Probing Classifiers](#probing-classifiers)
   - [TCAV](#tcav)
7. **Era 4 — Mechanistic Interpretability (2020–Present)**
   - [The Superposition Problem](#the-superposition-problem)
   - [Causal Tracing & Activation Patching](#causal-tracing--activation-patching)
   - [Sparse Autoencoders (SAEs)](#sparse-autoencoders-saes)
   - [Circuit Tracing & Attribution Graphs](#circuit-tracing--attribution-graphs)
   - [GoodFire (2026)](#goodfire-2026)
8. [Comparison Matrix](#comparison-matrix)
9. [Open Challenges](#open-challenges)
10. [Summary](#summary)
11. [References](#references)

---

## Motivation

For a long time, AI models have been viewed as impenetrable black boxes.
For more than a decade, several groundbreaking publications have allowed us to probe inside trained networks to understand why they behave the way they do. The insights we gain are crucial for mapping where our models shine and, more importantly, where they fall short. These interpretability tools are essential for improving AI systems, and every AI practitioner should be leveraging them. Spurred by Goodfire's recent announcements around frontier-scale feature steering, I’ve put together a brief history of the most important methods in Explainable AI.


## Why Does Interpretability Matter?

Deep neural networks achieve remarkable performance across nearly every domain of science and engineering. Yet they remain fundamentally opaque: a model can predict a patient's cancer risk, decide a loan application, or generate text indistinguishable from a human expert — all without exposing the reasoning behind its decision.

This opacity creates several interrelated problems:

**Safety & Alignment** — Black-box models may behave unpredictably in deployment, particularly under distribution shift. Understanding internal representations is essential for ensuring models act as intended, especially as AI systems become more capable.

**Scientific Discovery** — Models trained on biological or physical data encode patterns that may represent genuine scientific knowledge. Interpretability lets scientists extract new hypotheses — e.g., discovering novel regulatory motifs from genomics models.

**Debugging & Trust** — The ["clever Hans"](https://en.wikipedia.org/wiki/Clever_Hans) problem: a model may achieve high accuracy by exploiting spurious correlations (e.g., classifying "pneumonia risk" based on the hospital name). Interpretability is the go-to aproach to catch these failures before deployment.

**Regulatory Compliance** — GDPR's right-to-explanation, FDA AI/ML guidance for medical devices, and financial regulations (SR 11-7) increasingly require that automated decisions be explainable to affected individuals.

> **Key insight:** Interpretability is not a single thing. A regulator needs a legal-quality explanation; a scientist needs a falsifiable hypothesis; a safety researcher needs a causal account of model behavior. Different contexts demand different methods.

---

## A Taxonomy of Methods

The field organises methods along several independent axes:

**By Scope**
- *Local:* explains a single prediction — "why did the model output X for this input?"
- *Global:* explains model behaviour overall — "what features does this model rely on in general?"

**By Model Access**
- *Model-agnostic:* treats the model as a black box; queries via inputs and outputs only
- *Model-specific:* exploits the network architecture, weights, or gradients


**By Explanation Type**
- *Feature attribution:* which inputs (pixels, tokens, genes) matter most?
- *Concept-based:* which human-meaningful concepts activate?
- *Example-based:* which training samples are most influential?
- *Mechanistic:* what computational circuits produce the output?

---

## Historical Timeline

| Year | Milestone |
|------|-----------|
| 2013 | Saliency Maps (Simonyan et al.) |
| 2014 | Gradient × Input, Deconvolution Networks |
| 2015 | LRP (Bach et al.), CAM (Zhou et al.) |
| 2015 | Guided Backpropagation (Springenberg et al.) |
| 2016 | LIME (Ribeiro et al.), Grad-CAM (Selvaraju et al.) |
| 2016 | Probing Classifiers (Alain & Bengio) |
| 2017 | Integrated Gradients (Sundararajan et al.) |
| 2017 | DeepLIFT (Shrikumar et al.) |
| 2017 | SHAP (Lundberg & Lee) |
| 2018 | TCAV (Kim et al.) |
| 2019 | Attention-as-explanation debate |
| 2020 | TreeSHAP (Nature Machine Intelligence, Jan 2020); Transformer Circuits (Elhage et al.) |
| 2021 | Superposition Hypothesis (Anthropic) |
| 2022 | Causal Tracing / ROME (Meng et al.) |
| 2023 | Sparse Autoencoders (Cunningham et al., Bricken et al.) |
| 2024 | SAE scaling (Gao et al.); Circuit tracing |
| 2025–26 | Attribution graphs; GoodFire |

---

## Era 1 — Gradient-Based Methods (2013–2016)

*The first attempts to look inside neural networks using the machinery already present in backpropagation.*

---

### Saliency Maps

**Simonyan, Vedaldi & Zisserman, 2013** | *Gradient · Local*

The simplest idea in network interpretation: compute the gradient of the class score $S_c$ with respect to every pixel of the input image $\mathbf{x}$. A large absolute gradient at pixel $i$ means the model output is highly sensitive to small changes at that location — it is "salient."

$$\text{Saliency}_i = \left|\frac{\partial S_c(\mathbf{x})}{\partial x_i}\right|$$

*A single backward pass computes all pixel gradients simultaneously. Visualised as a heatmap over the input image.*

The procedure is straightforward: forward-pass the image, backpropagate from the target class neuron to the input layer, take absolute values. Bright regions in the resulting heatmap are the pixels the model is most locally sensitive to.

**Algorithm**

```text
function SaliencyMap(model, x, class_c):
    x.requires_grad = True
    score = model.forward(x)[class_c]       # forward pass
    score.backward()                         # single backward pass
    saliency = abs(x.grad)                   # pixel-wise absolute gradient
    return saliency                          # bright regions → high sensitivity
```

**Strengths**
- Extremely fast — a single backward pass
- Applicable to any differentiable model
- No hyperparameters required
- Provides immediate intuition about model focus

**Limitations & Failure Modes**
- *Saturation:* gradients vanish at flat regions of ReLU networks, missing genuinely important features that happen to be in saturated regimes
- *Noise:* raw gradient maps are high-frequency and visually incoherent
- *Symmetry problem:* treats positive and negative contributions equally (absolute value loses sign)
- *Not faithful:* reflects only the local linearisation, not global feature importance
- *Fails completeness:* attributions do not sum to the model output

> **Insight:** The critical "randomisation sanity check" (Adebayo et al., 2018) showed that saliency maps look nearly identical whether computed on a trained network or one with randomly initialised weights — suggesting they capture input structure, not model decisions.

*Simonyan, Vedaldi, Zisserman. "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps." arXiv:1312.6034, 2013.*

---

### Gradient × Input

**Baehrens, Schroeter, Harmeling, Kawanabe, Hansen & Müller, 2010; extended to deep CNNs ~2014** | *Gradient · Local*

The principle of multiplying the local gradient by the input value to produce explanation vectors was established by Baehrens et al. (2010) in JMLR ("How to explain individual classification decisions"), applied to SVMs, Parzen windows, and early neural networks — predating deep learning saliency work by several years. Deep learning researchers in ~2014 adapted this formulation to convolutional networks. The core idea: multiply the gradient by the actual input value, gating attribution by whether the feature is actually present — a feature with zero activation contributes zero attribution regardless of how sensitive the model is to it.

$$\phi_i(\mathbf{x}) = x_i \cdot \frac{\partial F(\mathbf{x})}{\partial x_i}$$

*Equivalent to a first-order Taylor expansion of $F(\mathbf{x})$ around the zero-input baseline.*

The intuition is that both conditions must hold for a feature to be "important": the model must be sensitive to it *and* the feature must actually be present in the input.

**Algorithm**

```text
function GradientXInput(model, x, class_c):
    x.requires_grad = True
    score = model.forward(x)[class_c]
    score.backward()                         # compute ∂score/∂x_i
    attribution = x * x.grad                 # element-wise: input × gradient
    return attribution                       # positive = excitatory, negative = inhibitory
```

**Strengths**
- One backward pass — no additional cost over vanilla gradients
- Sharper, cleaner maps than raw gradient magnitude
- Zero attribution for absent features (intuitively correct)
- Satisfies *implementation invariance*

**Limitations & Failure Modes**
- *Saturation still applies:* gradient still vanishes at saturated ReLU plateaus
- *Implicit baseline at zero:* not always meaningful outside image tasks
- *Fails sensitivity axiom:* can assign zero attribution to a feature that actually changes the output, if the gradient happens to be zero
- Integrated Gradients was designed explicitly to fix this deficiency

---

### Guided Backpropagation

**Springenberg, Dosovitskiy, Brox & Riedmiller, 2015** | *Gradient · Visualization*

Guided Backpropagation modifies standard gradient backpropagation through ReLU activations by applying a double mask: propagate only gradients that are both positive *and* originate from a positively activated neuron. This combines the "deconvolution" trick of Zeiler & Fergus with the standard gradient approach.

$$\frac{\partial f}{\partial x_i}^{\text{guided}} = \underbrace{\mathbb{1}\!\left[\frac{\partial f}{\partial h_i} > 0\right]}_{\text{deconv mask}} \cdot \underbrace{\mathbb{1}[x_i > 0]}_{\text{ReLU mask}} \cdot \frac{\partial f}{\partial h_i}$$

*Vanilla gradients apply neither mask. Deconvolution applies only the first. Guided Backpropagation applies both, yielding the sharpest visualisations.*

**Algorithm**

```text
function GuidedBackprop(model, x, class_c):
    # Register modified backward hooks on all ReLU layers
    for each ReLU layer in model:
        hook(grad_output):
            forward_mask  = (layer.input > 0)     # standard ReLU mask
            gradient_mask = (grad_output > 0)     # deconvolution mask
            return grad_output * forward_mask * gradient_mask

    x.requires_grad = True
    score = model.forward(x)[class_c]
    score.backward()                              # guided backward pass
    return x.grad                                 # sharp, positive-only map
```

**Strengths**
- Produces **visually sharp and clean** heatmaps
- Single backward pass — no overhead
- Combines naturally with Grad-CAM ("Guided Grad-CAM")

**Limitations & Failure Modes**
- *Not a true model explanation:* Sixt et al. (2020) showed it essentially reconstructs the input image, not what the model uses to decide
- *Class-insensitive:* maps are often very similar for different target classes
- *Model-blind:* produces nearly identical maps even for randomly initialised networks
- Fails the Adebayo et al. (2018) randomisation sanity checks

> **Insight:** Guided Backpropagation is one of the most widely-cited cautionary tales in interpretability. Its maps look convincing and sharp, yet are largely independent of the model's actual learned weights — demonstrating that visual plausibility is a poor proxy for faithfulness.

*Springenberg, Dosovitskiy, Brox, Riedmiller. "Striving for Simplicity: The All Convolutional Net." ICLR Workshop, 2015.*

---

### Layer-wise Relevance Propagation (LRP)

**Bach, Binder, Montavon, Klauschen, Müller & Samek, 2015** | *Propagation · Local*

LRP takes a fundamentally different approach: instead of computing derivatives, it redistributes the model's output score backward through the network layer by layer, governed by a *conservation of relevance* principle. The total relevance is never created or destroyed; it is simply re-allocated.

$$R_i^{(\ell)} = \sum_j \frac{a_i^{(\ell)}\, w_{ij}}{\sum_{i'} a_{i'}^{(\ell)}\, w_{i'j} + b_j + \varepsilon} \cdot R_j^{(\ell+1)} \qquad \text{(LRP-}\varepsilon\text{ rule)}$$

$a_i^{(\ell)}$ = activation of neuron $i$ at layer $\ell$; $w_{ij}$ = weight from $i$ to $j$; $b_j$ = bias of neuron $j$; $\varepsilon$ = small stabiliser. Including $b_j$ in the denominator is mandatory: omitting it violates the conservation property $\sum_i R_i^{(\ell)} = \sum_j R_j^{(\ell+1)} = f(\mathbf{x})$, as the bias's share of each neuron's activation would go unaccounted.

Different propagation rules trade off between stability and specificity: LRP-0 is exact but unstable near zero activations; LRP-ε adds a stabiliser; LRP-γ boosts positive contributions. In practice, different rules are applied to different layer types.

**Algorithm**

```text
function LRP_epsilon(model, x, class_c, epsilon=1e-6):
    # activations[layer][neuron] = forward-pass activation value at that neuron
    # weights[layer][i][j]       = weight from neuron i in layer to neuron j in layer+1
    # biases[layer+1][j]         = bias of neuron j in layer+1
    activations = model.forward_cache_all(x)
    weights     = model.get_weights()
    biases      = model.get_biases()

    R = {}
    R[output_layer] = zeros(n_output_neurons)
    R[output_layer][class_c] = model.output(x)[class_c]   # seed relevance at target

    for layer in range(n_layers - 1, 0, -1):              # from last hidden down to layer 1
        R[layer] = zeros(layer_size[layer])
        for neuron_i in range(layer_size[layer]):
            accumulated = 0.0
            for neuron_j in range(layer_size[layer + 1]):
                # Numerator: weighted contribution of neuron_i to neuron_j
                numerator = activations[layer][neuron_i] * weights[layer][neuron_i][neuron_j]
                # Denominator: full pre-activation of neuron_j including its bias
                # Omitting biases[layer+1][neuron_j] violates conservation
                denom_sum = sum(activations[layer][k] * weights[layer][k][neuron_j]
                                for k in range(layer_size[layer]))
                denominator = denom_sum + biases[layer + 1][neuron_j] + epsilon
                accumulated += (numerator / denominator) * R[layer + 1][neuron_j]
            R[layer][neuron_i] = accumulated

    # Conservation: sum(R[layer]) == sum(R[layer+1]) == model output, at every layer
    return R[1]    # input-layer relevance scores (one per input feature)
```

**Strengths**
- **Conservation principle** provides a theoretical foundation absent from gradient methods
- Completeness: pixel-level attributions sum to the model output
- Applicable to CNNs, RNNs, LSTMs, and Transformers
- Can separately track positive and negative contributions
- No baseline input required

**Limitations & Failure Modes**
- *Rule selection:* different propagation rules (LRP-0, LRP-ε, LRP-γ, LRP-z+) give significantly different results; the choice is not principled
- Requires layer-by-layer implementation — not truly model-agnostic
- Conservation alone does not guarantee that attributions are correct
- Under certain rules, LRP reduces to Gradient × Input (Kindermans et al., 2016)

> **Insight:** LRP is the dominant attribution method in the scientific/medical imaging community, where summing to the model output (completeness) is a hard requirement. The iNNvestigate and Captum libraries provide production-ready implementations.

*Bach, Binder, Montavon, Klauschen, Müller, Samek. "On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-wise Relevance Propagation." PLOS ONE, 2015.*

---

## Era 2 — Attribution & Model-Agnostic Methods (2016–2017)

*Axiomatic foundations, game-theoretic rigour, and the first truly model-agnostic approaches.*

---

### LIME

**Ribeiro, Singh & Guestrin, 2016** | *Model-Agnostic · Local*

LIME (Local Interpretable Model-agnostic Explanations) bypasses the need for model internals entirely. It approximates any black-box model *locally* around a specific input with a simple, interpretable model — typically a sparse linear classifier. The key insight is that even complex decision boundaries are approximately linear in a small neighbourhood.

$$\xi(\mathbf{x}) = \arg\min_{g \in \mathcal{G}}\; \mathcal{L}(f, g, \pi_{\mathbf{x}}) + \Omega(g)$$

$\pi_\mathbf{x}(z) = \exp(-D(\mathbf{x},z)^2/\sigma^2)$ weights samples by proximity; $\Omega(g)$ penalises complexity (e.g., L1 norm on coefficients). $\mathcal{L}$ measures local fidelity to the black-box $f$.

The algorithm: sample perturbed versions of the input in interpretable space (superpixels for images, words for text, cells for tabular data), query the model on each, weight by proximity, and fit a weighted sparse linear model. The linear coefficients are the explanations.

**Algorithm**

```text
function LIME(model, x, n_samples=1000, n_features=10, sigma=0.75):
    # sigma controls the width of the locality kernel
    segments = interpret_segments(x)           # superpixels / tokens / feature groups
    all_masks, all_outputs, all_weights = [], [], []

    for s in range(n_samples):
        # Randomly include/exclude each segment (1 = present, 0 = replaced by baseline)
        z_mask = random_binary_mask(num_segments=len(segments))
        z      = reconstruct_from_mask(x, z_mask, segments)
        y_hat  = model.predict(z)              # single black-box query

        # Exponential proximity kernel: inputs closer to x get higher weight
        dist  = euclidean_distance(x, z)
        w_s   = exp(-(dist ** 2) / (sigma ** 2))
        all_masks.append(z_mask)
        all_outputs.append(y_hat)
        all_weights.append(w_s)

    # Stage 1: discrete feature selection to find the top n_features segments
    # (uses forward selection or a separate Lasso pre-pass purely for subset selection)
    selected_indices = feature_selection(all_masks, all_outputs, all_weights,
                                         max_features=n_features)

    # Stage 2: fit weighted Ridge regression (L2 penalty) on the selected features
    # Ridge is used here — not Lasso — for numerical stability when features are correlated
    X_selected = [mask[selected_indices] for mask in all_masks]
    phi = weighted_ridge_regression(X=X_selected, y=all_outputs, weights=all_weights)
    return phi                                 # one coefficient per selected segment
```

**Strengths**
- **Truly model-agnostic** — works with any model, including non-differentiable ones
- Explanations are human-readable sparse linear models
- Works across modalities: tabular, text, images, time-series
- Straightforward to implement and deploy

**Limitations & Failure Modes**
- *Instability:* stochastic sampling makes different runs produce different explanations for the same input
- *Neighbourhood definition:* the perturbation strategy critically affects results; there is no universal best choice
- *Local only:* provides no global view of model behaviour
- *Spurious correlations:* local linear approximation may capture shortcuts rather than true model reasoning
- Expensive: requires many forward passes (typically 500–5000)

> **Insight:** LIME famously exposed a classifier that accurately identified elevated pneumonia risk in asthmatic patients — but for the wrong reason. The model had learned this from hospital admission patterns, not medical causation. Interpretability prevented a dangerous deployment.

*Ribeiro, Singh, Guestrin. "Why Should I Trust You?: Explaining the Predictions of Any Classifier." KDD, 2016.*

---

### Grad-CAM

**Selvaraju et al., 2017 | CAM: Zhou et al., 2016** | *Gradient · Visualization*

Class Activation Mapping (CAM) and its generalisation Grad-CAM address a specific goal: localising *where* in an image the model is looking when making a decision. CAM achieves this by exploiting the architecture of networks with a Global Average Pooling (GAP) layer before the classifier; Grad-CAM extends the idea to arbitrary CNN architectures using gradients as surrogate weights.

$$\alpha_k^c = \frac{1}{Z}\sum_i\sum_j \frac{\partial y^c}{\partial A_{ij}^k} \qquad L_{\text{Grad-CAM}}^c = \text{ReLU}\!\left(\sum_k \alpha_k^c\, A^k\right)$$

$A^k$ = feature map $k$ from the last convolutional layer; $\alpha_k^c$ = global-average-pooled gradient of class score $y^c$ w.r.t. map $k$; ReLU discards negative importance.

Guided Grad-CAM combines the high spatial resolution of Guided Backpropagation with the class-discriminativeness of Grad-CAM by element-wise multiplying the two maps.

**Algorithm**

```text
function GradCAM(model, x, class_c):
    # A[channel][row][col] = activation of channel at spatial position (row, col)
    A     = model.last_conv_layer_activations(x)   # shape: [n_channels, height, width]
    score = model.class_score(x, class_c)          # scalar class logit
    score.backward()                               # populate gradients

    # grads[channel][row][col] = gradient of score w.r.t. A[channel][row][col]
    grads = gradient_of(score, with_respect_to=A)

    # alpha[k] = mean gradient of channel k across all spatial positions (row, col)
    alpha = [mean(grads[k]) for k in range(n_channels)]

    # Weighted combination: sum over channels, keep only positive activations via relu
    heatmap = relu(sum(alpha[k] * A[k] for k in range(n_channels)))  # shape: [height, width]

    heatmap = bilinear_upsample(heatmap, target_size=x.shape)   # match input resolution
    return heatmap / max(heatmap)              # normalise to range [0, 1]
```

**Strengths**
- Works on any CNN without architectural modification
- **Class-discriminative** — different classes produce genuinely different heatmaps
- Fast: one forward + one backward pass
- Large ecosystem of extensions: Grad-CAM++, Score-CAM, Layer-CAM, EigenCAM

**Limitations & Failure Modes**
- *Coarse resolution:* heatmaps are limited to the spatial resolution of the last conv layer
- Highlights spatial regions, not individual pixels or features
- Limited to convolutional architectures; requires adaptation for Transformers
- Nohara et al. (2024) showed Grad-CAM and SHAP give divergent explanations on tabular data, raising faithfulness questions

*Selvaraju, Cogswell, Das, Vedantam, Parikh, Batra. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV, 2017.*

---

### DeepLIFT

**Shrikumar, Greenside & Kundaje, 2017** | *Propagation · Local*

DeepLIFT (Deep Learning Important FeaTures) directly addresses the saturation problem of gradient methods by propagating *differences from a reference* rather than instantaneous derivatives. Every neuron's contribution is measured as the difference between its actual activation and its activation on a chosen reference input $\mathbf{x}^0$.

$$\Delta x_i = x_i - x_i^0, \quad \Delta y = y - y^0, \quad \sum_i C_{\Delta x_i \Delta y} = \Delta y \quad \text{(Summation-to-delta)}$$

Contributions $C_{\Delta x_i \Delta y}$ are propagated layer by layer. Summation-to-delta is analogous to LRP's conservation principle but expressed in terms of differences from a reference.

Two propagation rules govern how contributions flow through activation functions:

**Rescale Rule** — When $\Delta x_i \neq 0$, the multiplier $m_{\Delta x_i \Delta t} = \Delta t / \Delta x_i$ is used in place of the gradient $\partial t / \partial x_i$. This avoids the 0/0 problem that arises when gradients vanish at saturation, while still being computable in a single backward pass.

**RevealCancel Rule** — Separately propagates positive and negative contributions through neurons. This reveals hidden cancellations: features that excite a neuron and features that inhibit it simultaneously, which would average out and disappear under gradient-based methods, are each made visible.

**Algorithm**

```text
function DeepLIFT(model, x, x_ref, class_c, epsilon=1e-7):
    # a[layer][neuron]     = activation for actual input x
    # a_ref[layer][neuron] = activation for reference input x_ref
    a     = model.forward_cache_all(x)
    a_ref = model.forward_cache_all(x_ref)

    # Total output difference to be distributed back through the network
    delta_y = model.output(x)[class_c] - model.output(x_ref)[class_c]
    C = {}
    C[output_layer] = zeros(n_output_neurons)
    C[output_layer][class_c] = delta_y

    for layer in range(n_layers - 1, 0, -1):
        C[layer] = zeros(layer_size[layer])
        for neuron_i in range(layer_size[layer]):
            # delta_a: how much did this neuron's activation change from reference?
            delta_a = a[layer][neuron_i] - a_ref[layer][neuron_i]
            # delta_t: corresponding change in the downstream target neuron
            delta_t = a[layer + 1] - a_ref[layer + 1]   # vector over layer+1 neurons

            if abs(delta_a) > epsilon:
                # Rescale rule: ratio of activation differences (avoids vanishing gradient)
                multiplier = delta_t / delta_a
            else:
                # Gradient fallback: use instantaneous gradient when delta_a is negligible
                multiplier = gradient_of(model, input=a[layer][neuron_i])
            C[layer][neuron_i] = sum(multiplier * C[layer + 1])

    # Completeness: sum(C[input_layer]) == delta_y
    return C[input_layer]
```

**Strengths**
- **Avoids saturation** — no vanishing gradients at ReLU plateaus
- Single backward pass through the network
- Separately tracks positive and negative contributions
- Completeness: attributions sum to $\Delta y$ from reference
- Forms the theoretical basis for DeepSHAP

**Limitations & Failure Modes**
- *Reference sensitivity:* results depend heavily on the choice of baseline (zero input, mean input, blurred input)
- Architecture-specific: requires separate implementation for each layer type (conv, LSTM, etc.)
- RevealCancel rule is not well-defined for all activation functions
- Less formally principled than the Shapley-value framework

> **Insight:** DeepLIFT became the workhorse of genomics interpretability. In models like Basset, Enformer, and Sei, the reference genome provides a natural, biologically meaningful baseline — and the differential contributions directly answer "what makes this sequence different from a typical one?"

*Shrikumar, Greenside, Kundaje. "Learning Important Features Through Propagating Activation Differences." ICML, 2017.*

---

### Integrated Gradients

**Sundararajan, Taly & Yan, 2017** | *Gradient · Axiomatic*

Integrated Gradients (IG) accumulates gradients along the straight-line path from a baseline $\mathbf{x}'$ to the actual input $\mathbf{x}$. Where gradient × input uses only the instantaneous derivative at $\mathbf{x}$ (with its saturation problems), IG integrates over the entire path, capturing the full contribution of each feature.

$$\text{IG}_i(\mathbf{x}) = (x_i - x_i') \times \int_0^1 \frac{\partial F\!\left(\mathbf{x}' + \alpha(\mathbf{x}-\mathbf{x}')\right)}{\partial x_i}\, d\alpha$$

In practice, the integral is approximated with a Riemann sum over $m \approx 50$–$300$ steps.

What makes IG theoretically compelling is that it uniquely satisfies two axioms that any attribution method should obey:

- **Sensitivity:** If $F(\mathbf{x}) \neq F(\mathbf{x}')$ but the inputs differ only in feature $i$, then $\text{IG}_i(\mathbf{x}) \neq 0$. The method cannot miss a feature that demonstrably changed the output.
- **Implementation Invariance:** Two functionally equivalent networks (computing the same input-output map via different architectures) receive identical attributions. Attribution should describe the *function*, not the implementation.

Additionally, IG satisfies **completeness**: $\sum_i \text{IG}_i(\mathbf{x}) = F(\mathbf{x}) - F(\mathbf{x}')$ — all attributions sum exactly to the output difference from baseline.

**Algorithm**

```text
function IntegratedGradients(model, x, x_baseline, class_c, n_steps=50):
    # Approximate the path integral by summing gradients at n_steps equally-spaced points
    # between x_baseline and x along a straight line
    grad_sum = zeros_like(x)

    for k in range(1, n_steps + 1):
        step_fraction = k / n_steps                       # ranges from 1/n_steps to 1.0
        x_interp      = x_baseline + step_fraction * (x - x_baseline)
        x_interp.requires_grad = True
        score = model(x_interp)[class_c]
        score.backward()
        grad_sum += gradient_of(score, with_respect_to=x_interp)

    # Riemann-sum approximation of the integral; multiply by (x - baseline) per feature
    IG = (x - x_baseline) * (grad_sum / n_steps)

    # Completeness check: sum(IG) should approximately equal model(x)[c] - model(x_baseline)[c]
    return IG
```

**Strengths**
- **Axiomatic foundation** — uniquely satisfies Sensitivity + Implementation Invariance
- Completeness guaranteed
- Works on any differentiable model (vision, NLP, graphs, tabular)
- No architectural modification needed
- Widely adopted in industry (Google, Anthropic, Meta)

**Limitations & Failure Modes**
- *Baseline choice:* results depend on $\mathbf{x}'$; no universally correct choice exists
- *Computationally expensive:* requires $m$ forward passes (typically 50–300)
- Linear interpolation path may not follow the natural data manifold
- Noisy on high-dimensional inputs; SmoothGrad averaging sometimes needed

> **Insight:** Expected Gradients (GradSHAP) extends IG by replacing the single fixed baseline with an expectation over training data: $\phi_i \approx \mathbb{E}_{\mathbf{x}'}[(x_i - x_i') \partial F / \partial x_i|_{\mathbf{x}'}]$. This both eliminates baseline sensitivity and produces a principled connection to Shapley values.

*Sundararajan, Taly, Yan. "Axiomatic Attribution for Deep Networks." ICML, 2017.*

---

### SHAP, GradSHAP & DeepSHAP

**Lundberg & Lee, 2017** | *Model-Agnostic · Game Theory*

SHAP (SHapley Additive exPlanations) grounds feature attribution in cooperative game theory. Each input feature is treated as a "player" in a coalition game; its Shapley value is its *average marginal contribution* when added to all possible subsets of other features.

$$\phi_i(f) = \sum_{S \subseteq \mathcal{F} \setminus \{i\}} \frac{|S|!\;(|\mathcal{F}|-|S|-1)!}{|\mathcal{F}|!} \left[f_{S\cup\{i\}}(\mathbf{x}) - f_S(\mathbf{x})\right]$$

$f_S(\mathbf{x})$ = expected model output given only features in $S$, marginalising over the remaining features.

The Shapley value is the *unique* attribution satisfying four axioms: **Efficiency** ($\sum_i \phi_i = f(\mathbf{x}) - \mathbb{E}[f]$), **Symmetry** (equal-contribution features get equal $\phi$), **Dummy** (non-contributing features get $\phi_i = 0$), and **Linearity** ($\phi_i$ is linear in $f$).

Exact Shapley values require $O(2^n)$ model evaluations. Three practical variants address this:

**KernelSHAP** — Model-agnostic. Approximates Shapley values by solving a specially weighted least-squares regression over randomly sampled feature subsets. Slow but universally applicable.

**DeepSHAP** — Combines DeepLIFT's layer-by-layer propagation rules with multiple background reference samples to approximate Shapley values for deep networks. Fast and architecture-aware, but approximate.

**TreeSHAP (Lundberg et al., Nature Machine Intelligence, 2020)** — Exact Shapley values for tree-based models (XGBoost, LightGBM, random forests) in $O(TLD^2)$ — polynomial rather than exponential time. Formally published in *Nature Machine Intelligence* in January 2020; enables global feature importance for gradient-boosted trees.

**SHAP-XRT (Teneggi, Bharti, Romano & Sulam, 2022)** augments SHAP attributions with formal Conditional Independence Testing (CIT) via a randomised tree approach to determine whether each Shapley value is statistically significantly different from zero — distinguishing genuine feature importance from sampling noise.

**Algorithm**

```text
function KernelSHAP(model, x, x_bg, n_samples=1000):
    n_features = number_of_features(x)
    masks, outputs, weights = [], [], []

    for s in range(n_samples):
        # z_mask[i] = 1 means feature i is "present"; 0 means replaced by background
        z_mask  = random_binary_mask(n_features)
        z       = where(z_mask == 1, x, x_bg)
        f_z     = model.predict(z)

        # coalition_size = number of features included (i.e., set to 1) in this sample
        coalition_size = sum(z_mask)
        # Shapley kernel weight: assigns higher weight to small and large coalitions
        if coalition_size == 0 or coalition_size == n_features:
            w_s = large_constant               # handle degenerate coalitions
        else:
            binom = binomial_coefficient(n_features, coalition_size)
            w_s   = (n_features - 1) / (binom * coalition_size * (n_features - coalition_size))
        masks.append(z_mask); outputs.append(f_z); weights.append(w_s)

    # Efficiency constraint: phi values must sum to model(x) - mean model output
    expected_output = mean([model.predict(x_bg_sample) for x_bg_sample in background_set])
    phi = weighted_least_squares(
              X=masks,
              y=outputs,
              weights=weights,
              sum_constraint=model.predict(x) - expected_output)
    return phi                                 # one Shapley value approximation per feature

function TreeSHAP(tree_ensemble, x):
    # Exact Shapley values in polynomial time: O(n_trees * n_leaves * max_depth^2)
    all_contributions = zeros(n_features)
    for tree in tree_ensemble:
        # Dynamic programming: track which features are "in the coalition" at each split node
        contributions = compute_path_contributions(tree, x)   # O(n_leaves * max_depth^2)
        all_contributions += contributions
    return all_contributions                   # exact Shapley value per feature
```

**Strengths**
- **Game-theoretically principled** — unique attribution satisfying four axioms
- Unifies LIME, DeepLIFT, and LRP as special cases under the Shapley framework
- Can aggregate $\phi_i$ across examples for global feature importance
- Widely deployed ecosystem (Python `shap` library, 20M+ downloads)
- TreeSHAP is exact and fast for gradient-boosted trees

**Limitations & Failure Modes**
- *Exponential cost* for exact computation: $O(2^n)$ feature subsets
- *Marginal vs. conditional:* marginalising features independently assumes independence; real feature correlations are violated
- KernelSHAP requires hundreds to thousands of model calls — slow for deep networks
- DeepSHAP and GradSHAP are approximations, not exact Shapley values
- Slack et al. (2020) showed adversarial models can produce plausible-looking SHAP explanations that hide discriminatory decision-making

*Lundberg, Lee. "A Unified Approach to Interpreting Model Predictions." NeurIPS, 2017. | Teneggi, Bharti, Romano, Sulam. "SHAP-XRT: The Shapley Value Meets Conditional Independence Testing." arXiv:2207.07038, 2022. | Frye et al. "Asymmetric Shapley Values." NeurIPS, 2020.*

---

## Era 3 — Concept-Based Methods (2016–2020)

*Moving beyond individual pixels and tokens to human-understandable concepts.*

---

### Probing Classifiers

**Alain & Bengio, 2016** | *Diagnostic · Linear*

Probing classifiers ask: what information is linearly accessible in a model's internal representations? A simple (usually linear) classifier is trained on frozen activations at each layer to predict some target property — part-of-speech tags, syntactic depth, named-entity labels, or any concept of interest.

$$\hat{y} = \text{softmax}\!\left(\mathbf{W}\, \mathbf{h}^{(\ell)}(\mathbf{x}) + \mathbf{b}\right)$$

$\mathbf{h}^{(\ell)}(\mathbf{x})$ = frozen representation at layer $\ell$; $\mathbf{W}, \mathbf{b}$ are the only trained parameters. High accuracy at layer $\ell$ implies the concept is linearly decodable from representations at that depth.

Probing studies of BERT (Tenney et al., 2019) produced one of the canonical results in NLP interpretability: syntactic structure (POS tags, parse depth) is encoded in early layers 1–4; semantic relationships (coreference, semantic roles) emerge in middle layers 5–9; the final layers are most task-specific. The model implicitly rediscovers the classical NLP pipeline.

**Algorithm**

```text
function ProbingClassifier(model, labelled_dataset, concept, target_layer_indices):
    # For each layer, train a small linear classifier to predict the concept label
    # from the frozen activations at that layer
    accuracy_per_layer = {}

    for layer_idx in target_layer_indices:
        representations, labels = [], []
        for (x, concept_label) in labelled_dataset:
            # Extract activations without updating model weights
            h = model.get_activations(x, layer_index=layer_idx)
            representations.append(h)
            labels.append(concept_label)

        # Train only W and b; model weights remain frozen throughout
        W, b  = train_logistic_regression(X=representations, y=labels)
        acc   = evaluate_accuracy(W, b, validation_set)
        accuracy_per_layer[layer_idx] = acc

    # High accuracy at layer_idx means the concept is linearly decodable there
    # Use MDL probing (Voita & Titov, 2020) to control for probe overfitting
    return accuracy_per_layer
```

**Strengths**
- Simple to implement — only a linear classifier is trained
- Reveals *where* across layers information becomes available
- Cheap: only needs frozen activations, not model gradients
- Applicable to any architecture

**Limitations & Failure Modes**
- *Correlation ≠ causation:* high probe accuracy means the information is present, not that the model uses it for its decisions
- *Probe complexity:* a powerful nonlinear probe can learn the task from noise, giving spuriously high accuracy
- *No directionality:* probing does not show how a concept influences the output
- MDL probes (Voita & Titov, 2020) and control tasks (Hewitt & Liang, 2019) are needed to guard against overfitting artifacts

*Alain, Bengio. "Understanding Intermediate Layers Using Linear Classifier Probes." ICLR Workshop, 2017. | Tenney et al. "BERT Rediscovers the Classical NLP Pipeline." ACL, 2019.*

---

### TCAV

**Kim, Wattenberg, Gilmer et al., 2018** | *Concept-Based · Global*

Testing with Concept Activation Vectors (TCAV) lets domain experts define concepts in natural terms (e.g., "striped texture," "young patient," "presence of artifact") and test whether the model actually uses those concepts when making class-$c$ predictions.

The method trains a linear classifier in a model's activation space to detect the concept, then measures the directional derivative of the class score along the concept direction.

$$S_{C,k,l}^c(\mathbf{x}) = \nabla h_{l,k}(f_l(\mathbf{x})) \cdot \mathbf{v}_l^C \qquad \text{TCAV}^c = \frac{|\{x \in X_k : S_{C,k,l}^c(x) > 0\}|}{|X_k|}$$

$\mathbf{v}_l^C$ = Concept Activation Vector (normal to the linear decision boundary in layer-$l$ activation space); $S$ = directional derivative of the class logit along $\mathbf{v}_l^C$; TCAV = fraction of class-$c$ examples positively influenced by the concept. A t-test across random negatives provides statistical significance.

**Algorithm**

```text
function TCAV(model, concept_examples, random_examples, class_c, layer_index):
    # Step 1: learn Concept Activation Vector (CAV) in the layer's activation space
    # CAV = the normal vector of a linear classifier boundary separating concept from random
    h_pos = [model.get_activations(x, layer_index) for x in concept_examples]
    h_neg = [model.get_activations(x, layer_index) for x in random_examples]
    v_C   = train_linear_svm(positives=h_pos, negatives=h_neg).decision_boundary_normal

    # Step 2: TCAV score = fraction of class-c inputs that are positively influenced
    # by the concept (directional derivative along v_C is positive)
    positive_count = 0
    for x in test_examples_of_class_c:
        h = model.get_activations(x, layer_index)
        # Directional derivative: how much would the class logit increase
        # if we nudged activations in the direction of the concept vector?
        grad_h = gradient_of(model.class_logit(x, class_c), with_respect_to=h)
        sensitivity_score = dot_product(grad_h, v_C)
        if sensitivity_score > 0:
            positive_count += 1

    TCAV_score = positive_count / len(test_examples_of_class_c)

    # Step 3: check statistical significance against randomly generated CAVs
    null_scores = [compute_tcav_with_random_cav(model, class_c, layer_index)
                   for _ in range(n_random_baselines)]
    p_value = t_test(observed=TCAV_score, null_distribution=null_scores)
    return TCAV_score, p_value
```

**Strengths**
- Explanations in **human-understandable vocabulary**, not pixels
- No ML expertise required to define concepts — just provide example images or text
- Statistical testing gives confidence levels for each concept-class association
- Global: summarises concept importance across many examples, not just one
- Concepts need not correspond to input features (e.g., style, texture, demographic)

**Limitations & Failure Modes**
- *Concept example collection:* requires curated positive/negative sets for each concept
- CAV relies on linear separability of the concept in activation space
- Concepts must be predefined — no discovery of *novel* unanticipated concepts
- Results are sensitive to the layer chosen for CAV training
- Measures correlation of concept with class prediction, not causal influence

> **Insight:** TCAV directly inspired Concept Bottleneck Models (Koh et al., 2020), which force human-specified concepts to be explicit intermediate representations. This enables test-time concept interventions: "if I tell the model this patient does NOT have concept X, does its prediction change?"

*Kim, Wattenberg, Gilmer, Cai, Wexler, Viegas, Sayres. "Interpretability Beyond Classification Error: Applying TCAV." ICML, 2018.*

---

## Era 4 — Mechanistic Interpretability (2020–Present)

*Moving from post-hoc explanation to reverse-engineering the computation itself.*

---

### The Superposition Problem

**Elhage et al., 2022** | *Foundation*

Before describing modern mechanistic interpretability methods, it is necessary to understand what makes neural network interpretation fundamentally hard. The answer, established formally by Elhage et al. in their "Toy Models of Superposition," is **polysemanticity**: individual neurons in large language models respond to multiple unrelated concepts simultaneously.

Why does this happen? A network with $n$ neurons can represent at most $n$ orthogonal directions — but it needs to encode far more than $n$ features to model the world. High-dimensional spaces contain exponentially many nearly-orthogonal directions. Networks exploit this by encoding features as superpositions of neurons, at the cost of small but non-zero interference between features.

$$\mathbf{x} \approx \mathbf{W}^\top \mathbf{W}\, \mathbf{x} \qquad \|\mathbf{W}_i\|^2 \approx \text{importance}(i) \cdot \bigl(1 - S_i\bigr)$$

Features are encoded with weight proportional to their importance and inversely proportional to their sparsity $S_i$. Rare but important features are encoded precisely; common low-value features may be partially suppressed.

The implications are severe: a 512-neuron layer may encode 4,000+ features simultaneously. Neuron-level probes and gradient-based attributions are fundamentally limited because they attribute to neurons, not features. This motivates the shift to dictionary learning approaches.

*Elhage et al. "Toy Models of Superposition." Transformer Circuits Thread, 2022.*

---

### Causal Tracing & Activation Patching

**Meng, Bau, Andonian & Belinkov, 2022** | *Causal · Mechanistic*

Causal tracing is the first method in this review that makes genuinely *causal* rather than merely correlational claims about model internals. The central idea: identify which internal components causally determine an output by *patching* activations — restoring the activations from a clean run into a corrupted run and measuring how much the output recovers.

**Three-Phase Algorithm:**

1. **Phase 1 (Clean run):** forward pass on the original prompt $\mathbf{x}$; cache all activations $\{h_i^{(\ell)}\}$ at every site $(i, \ell)$
2. **Phase 2 (Corrupt run):** forward pass on a corrupted prompt $\tilde{\mathbf{x}}$ (e.g., add Gaussian noise to the subject token embeddings); record the degraded output probability
3. **Phase 3 (Patch run):** run the corrupted prompt but restore the clean activation $h_i^{(\ell)}$ at one specific site $(i, \ell)$; measure how much the output probability recovers

A site with high *average indirect effect* (AIE) — large recovery of output probability — is causally necessary for the behaviour. Meng et al. applied this to GPT-2 and GPT-J and discovered that factual associations ("The Eiffel Tower is in [Paris]") are localised to specific MLP layers in the middle third of the network, enabling the ROME (Rank-One Model Editing) method for targeted knowledge updates.

**Algorithm**

```text
function CausalTracing(model, x_clean, subject_positions, target_token):
    # Phase 1: run model on original (clean) input; cache every activation
    # h_clean[layer_idx][token_idx] = activation vector at that site
    h_clean  = model.forward_cache_all(x_clean)
    p_clean  = model.output_prob(target_token)   # probability assigned to target token

    # Phase 2: corrupt run — add Gaussian noise to embeddings of subject tokens only
    x_corrupt = add_gaussian_noise(x_clean, positions=subject_positions)
    h_corrupt = model.forward_cache_all(x_corrupt)
    p_corrupt = model.output_prob(target_token)  # degraded probability

    # Phase 3: for each (layer, token) site, restore the clean activation and measure
    # how much of the lost probability is recovered (= Average Indirect Effect, AIE)
    AIE = zeros(n_layers, n_tokens)
    for layer_idx in range(n_layers):
        for token_idx in range(n_tokens):
            p_patch = model.forward_with_patch(
                          input=x_corrupt,
                          patch_layer=layer_idx,
                          patch_token=token_idx,
                          patch_value=h_clean[layer_idx][token_idx]
                      ).output_prob(target_token)
            # AIE = probability recovery; high value = this site is causally necessary
            AIE[layer_idx][token_idx] = p_patch - p_corrupt

    # Total cost: n_layers * n_tokens additional forward passes
    return AIE
```

**Strengths**
- **Causal, not correlational** — establishes that a component is actually computing the relevant output
- Produces falsifiable, testable claims about model internals
- Enables downstream applications: model editing, circuit discovery, safety interventions
- Works at multiple granularities: token positions, layers, attention heads, MLP layers

**Limitations & Failure Modes**
- *Expensive:* requires one patching run per site — thousands of forward passes
- *Corruption choice matters:* Gaussian noise vs. different subject tokens produce divergent results (Hernandez et al., 2023)
- Can be misleading when model behaviour is distributed or redundant across circuits
- *Attribution patching* (linearised approximation) trades accuracy for speed but introduces approximation error
- Identifies *where* a computation happens, not *how*

*Meng, Bau, Andonian, Belinkov. "Locating and Editing Factual Associations in GPT." NeurIPS, 2022.*

---

### Sparse Autoencoders (SAEs)

**Cunningham et al., 2023 | Bricken et al., 2023** | *Feature Decomposition · Unsupervised*

Large language models contain far fewer neurons than there are concepts in the world. To compensate, networks compress information: rather than dedicating Neuron #502 exclusively to "dogs," a concept is encoded as a *combination* of activations — for example, "dog" might be represented by Neurons #10, #502, and #9000 firing together in a specific pattern. This compression makes individual neurons **polysemantic**: a single neuron may fire for French text, DNA sequences, *and* Python code — three completely unrelated things. Examining the model neuron-by-neuron produces noise, not insight.

A Sparse Autoencoder is a secondary neural network trained to act as a *translator* for the LLM. Its job is to observe the dense, tangled, polysemantic activations and project them into a far larger, untangled space where each feature corresponds to exactly one concept (**monosemanticity**). The mechanism has three stages:

1. **Input:** The LLM is frozen and fed large amounts of text. Activation vectors (the model's "thoughts") at a specific layer are recorded — e.g., a 512-dimensional vector $\mathbf{h}$.
2. **Encoder (Expansion):** The SAE projects that dense vector into an *overcomplete* space (e.g., 16,000 dimensions), separating the compressed concepts into many more independent dimensions.
3. **Decoder (Reconstruction):** The SAE compresses the expanded vector back to the original dimensionality, trying to perfectly reconstruct the original LLM activation.

Formally, this corresponds to learning a **sparse, overcomplete dictionary**: if superposition allows a layer with $d_\text{model}$ neurons to encode $d_\text{feat} \gg d_\text{model}$ features, training an overcomplete autoencoder with a sparsity penalty can recover those features.

$$\mathbf{f} = \text{ReLU}(\mathbf{W}_e\, \mathbf{h} + \mathbf{b}_e) \qquad \hat{\mathbf{h}} = \mathbf{W}_d\, \mathbf{f} + \mathbf{b}_d$$

$$\mathcal{L} = \underbrace{\|\mathbf{h} - \hat{\mathbf{h}}\|_2^2}_{\text{reconstruction}} + \underbrace{\lambda\, \|\mathbf{f}\|_1}_{\text{sparsity penalty}}$$

$\mathbf{W}_e \in \mathbb{R}^{d_\text{feat} \times d_\text{model}}$, $\mathbf{W}_d \in \mathbb{R}^{d_\text{model} \times d_\text{feat}}$, with $d_\text{feat} \gg d_\text{model}$ (overcomplete). Decoder columns are constrained to unit norm to prevent scale collapse. The $\ell_1$ penalty encourages only a few features to be active per token.

Bricken et al. (2023, Anthropic) applied this to a single-layer transformer's MLP and found that a 512-neuron layer encodes 4,096 interpretable, monosemantic features — each responding to a single coherent concept: "DNA sequences," "Hebrew text," "HTTP server responses," "legal documents," "base64 encoding," and thousands more.

Cunningham et al. (2023) independently applied SAEs to GPT-2 and demonstrated that the recovered features are both interpretable and causally active in model behaviour — features can be individually ablated or amplified to predictably change model outputs.

**Architecture choices & the dead neuron problem:**
- *Expansion factor* $d_\text{feat}/d_\text{model} \in [4\times, 32\times]$: larger dictionaries recover more features but require more compute
- *Where to apply:* residual stream, MLP post-activation, or attention output — each captures different aspects of computation
- *Dead neurons:* many SAE features become permanently inactive during training. Fixes include auxiliary loss terms, TopK activation (Gao et al., 2024 — forces exactly $k$ features active per token), and JumpReLU SAEs
- *Automated labelling:* a second LLM (Bills et al., 2023) can automatically identify what concept each feature encodes by examining its top-activating tokens

**Algorithm**

```text
function TrainSAE(model, activation_corpus, d_feat, lambda_sparse, n_steps):
    # d_feat >> d_model: overcomplete dictionary (typically 4x to 32x the model width)
    # W_enc[feature_idx][model_dim] = encoder weight matrix
    # W_dec[model_dim][feature_idx] = decoder weight matrix (columns must be unit-norm)
    W_enc = random_init(shape=[d_feat, d_model])
    W_dec = normalise_columns(random_init(shape=[d_model, d_feat]))
    b_enc = zeros(d_feat)
    b_dec = zeros(d_model)

    for step in range(n_steps):
        h = sample_batch(activation_corpus)    # frozen model activations, shape [batch, d_model]

        # Encode: project to overcomplete space; ReLU enforces non-negative, sparse activations
        pre_activation = matmul(W_enc, h) + b_enc      # shape [batch, d_feat]
        f = relu(pre_activation)                        # sparse feature activations

        # Decode: reconstruct original layer activations from feature activations
        h_hat = matmul(W_dec, f) + b_dec               # shape [batch, d_model]

        # Reconstruction loss: mean squared error over all elements
        recon_loss  = mean(sum_of_squares(h - h_hat))
        # Sparsity penalty: L1 norm encourages most features to be inactive per token
        sparse_loss = lambda_sparse * mean(sum(abs(f)))
        loss = recon_loss + sparse_loss

        loss.backward()
        optimizer.step(W_enc, W_dec, b_enc, b_dec)
        W_dec = normalise_columns(W_dec)               # re-enforce unit-norm on decoder columns

    return W_enc, W_dec    # W_enc rows = d_feat learned feature directions in activation space

function LabelFeature(SAE, activation_corpus, feature_idx, LLM):
    # Find which input tokens most strongly activate this feature
    top_tokens = find_max_activating_tokens(W_enc_row=SAE.W_enc[feature_idx],
                                             corpus=activation_corpus)
    return LLM.complete("What concept do these tokens share?\n" + top_tokens)
```

**Strengths**
- **Directly addresses superposition** — recovers genuine monosemantic features
- Unsupervised — no need to predefine concepts or provide labelled data
- Scalable: successfully applied to GPT-4-class models with 34M+ features (Anthropic, 2024)
- Features can be causally intervened on for steering and safety research
- Enables circuit-level analysis: features can be used as nodes in a computational graph

**Limitations & Failure Modes**
- *Feature interpretation remains laborious* — each of millions of features must be manually or automatically labelled
- Training is expensive: requires large numbers of model activations and significant compute
- Fundamental sparsity–reconstruction tradeoff: higher sparsity → more interpretable features but worse reconstruction
- Not all features are interpretable even after training; some remain polysemantic or abstract
- Features may reflect SAE training artifacts rather than genuine model representations

*Cunningham, Ewart, Riggs, Huben, Sharkey. "Sparse Autoencoders Find Highly Interpretable Features in Language Models." ICLR 2024 (arXiv:2309.08600). | Bricken, Templeton, Batson et al. "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning." Anthropic, 2023.*

---

### Circuit Tracing & Attribution Graphs

**Lindsey et al., 2025** | *Mechanistic · Causal*

If Sparse Autoencoders gave us the *dictionary* of the neural network's mind — isolating individual concepts — Circuit Tracing gives us the **wiring diagram**. Knowing a model has a "Paris" feature and an "Eiffel Tower" feature is useful, but it doesn't explain how the model connects them to answer a question. Circuit tracing bridges the gap between identifying static concepts and understanding dynamic computation, producing **attribution graphs** — directed computational graphs showing which features causally compute which outputs.

**1. The Cross-Layer Alignment Problem**

In a deep neural network, computation flows sequentially across dozens of layers. If SAEs are trained independently on Layer 4 and Layer 5, a translation problem arises: the concept "dogs" might be Feature #1042 in Layer 4 but Feature #8911 in Layer 5. Because these feature spaces don't naturally align, mapping how a thought flows from one layer to the next is unreliable. Lindsey et al. (2025) solved this with **Cross-Layer Transcoders (CLTs)**, which map the *input* activations of a layer directly to its *output* activations — aligning adjacent feature spaces by construction and providing explicit directed computational edges between layers.

**2. The Scalpel: Causal Activation Patching**

Once features are aligned across layers, researchers prove that Feature A *causes* Feature B to fire using **activation patching** — a controlled knockout experiment:

- **Clean run:** Feed the model a prompt and record all feature activations normally.
- **Intervention (ablation):** Run the model again, but artificially zero out a specific feature in an early layer (e.g., the "John" feature).
- **Measurement:** Observe downstream changes. If the "Indirect Object" feature drops by $\Delta$, a causal link is established.

Automating this process across thousands of feature pairs yields the causal effect of every source feature on every target feature, forming the edges of the attribution graph.

**3. Building the Attribution Graph**

The result is an **attribution graph** — a directed computational map of how the model "thinks": nodes are interpretable SAE features (e.g., "duplicate token head," "inhibition feature," "gender pronoun"); edges are causal weights verified by patching. For the *Indirect Object Identification* task ("John gave Mary a gift; she gave it to ___"), researchers found a literal circuit: early features identify all names in the sentence; *duplicate token* features note that "she" refers back to "Mary"; *S-inhibition* features actively suppress "John" (the subject) from becoming the output; the final layers compile this into the output token "John". Lindsey et al. (2025) traced planning circuits, colour-concept binding, and multimodal feature structures in frontier models.

**Algorithm**

```text
function CircuitTracing(model, feature_extractor, x, behaviour):
    # feature_extractor can be a per-layer SAE or a Cross-Layer Transcoder (CLT).
    # Lindsey et al. (2025) used CLTs: each CLT[layer] maps
    #   input activations of layer -> output activations of layer,
    # providing explicit directed computational edges between adjacent layers.

    # Step 1: identify which features are active at each layer for this input
    # active_features[layer_idx] = list of feature indices with above-threshold activation
    active_features = {}
    for layer_idx in range(n_layers):
        h = model.get_activations(x, layer_index=layer_idx)
        f = feature_extractor[layer_idx].encode(h)   # sparse feature activations
        active_features[layer_idx] = top_k_active(f, k=50)

    # Step 2: attribution patching — estimate the causal effect of each source feature
    # on each target feature at the next layer
    edges = {}
    for layer_idx in range(1, n_layers):
        for feat_j in active_features[layer_idx]:
            for feat_i in active_features[layer_idx - 1]:
                # Zero out feature feat_i at layer layer_idx-1 and measure
                # the resulting change in feat_j activation at layer layer_idx
                delta = ablate_and_measure(
                            model, feature_extractor, x,
                            source_layer=layer_idx - 1, source_feature=feat_i,
                            target_layer=layer_idx,   target_feature=feat_j)
                if abs(delta) > threshold:
                    key = ((layer_idx - 1, feat_i), (layer_idx, feat_j))
                    edges[key] = delta

    # Step 3: build directed graph and verify each edge with ablation experiments
    graph = DirectedGraph(nodes=active_features, edges=edges)
    for edge in list(graph.edges):
        if not ablation_confirms_causal_role(model, x, edge, behaviour):
            graph.remove(edge)

    return graph   # causally verified feature-level computation graph
```

**Strengths**
- Produces the most complete mechanistic account of any method in this review
- Causal — each edge is verified by intervention experiments
- Operates at the level of interpretable semantic features, not raw neurons
- Enables targeted model editing and safety research at the circuit level

**Limitations & Failure Modes**
- Extremely expensive: requires SAE training + many patching experiments
- Circuits found may not be unique — redundant circuits often exist in parallel
- Does not yet scale to complex multi-step reasoning in frontier models
- Feature labelling bottleneck: circuits are only as interpretable as their component features

*Lindsey et al. "On the Biology of a Large Language Model." Anthropic, 2025. | Wang et al. "Interpretability in the Wild: A Circuit for IOI in GPT-2 small." ICLR, 2023.*

---

### GoodFire (2026)

**GoodFire Research, 2026** | *Feature Steering · Applied MI*

GoodFire represents the productisation of mechanistic interpretability. Building on SAE infrastructure developed by Anthropic and EleutherAI, GoodFire scales feature extraction and steering to frontier-scale models, enabling **real-time feature-level control** of language model behaviour — modifying SAE feature activations directly to steer model outputs in targeted, interpretable ways.

**Key Capabilities:**
- Extract and catalogue millions of semantic features from large production LLMs
- Steer model behaviour by amplifying or suppressing specific features at inference time
- Automated feature discovery and LLM-based labelling pipeline
- Cross-model feature comparison to understand architectural differences

**Applications Demonstrated:**
- Targeted debiasing without full fine-tuning
- Behaviour elicitation: surfacing suppressed or low-probability knowledge
- Safety-relevant feature identification and monitoring
- Interpretable ablation studies at the feature level


---

## Summary

As agentic AI becomes a staple in day-to-day model development, leveraging explainable AI will become seamless. Rather than an afterthought, XAI is poised to become an inherent part of model design, evaluation, and benchmarking, with AI agents helping practitioners deeply understand the data driving their networks. This opens an exciting next chapter where understanding model behavior actively guides the training process instead of just diagnosing it at the end. Biomedical applications, in particular, stand to benefit immensely from this shift, highlighting an urgent need for XAI tools designed specifically with complex biological and medical data in mind.


---

## References

1. **Simonyan, Vedaldi, Zisserman** (2013). Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps. *arXiv:1312.6034*

1b. **Baehrens, Schroeter, Harmeling, Kawanabe, Hansen, Müller** (2010). How to explain individual classification decisions. *Journal of Machine Learning Research (JMLR)*

2. **Zeiler, Fergus** (2014). Visualizing and Understanding Convolutional Networks. *ECCV*

3. **Bach, Binder, Montavon, Klauschen, Müller, Samek** (2015). On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-wise Relevance Propagation. *PLOS ONE*

4. **Springenberg, Dosovitskiy, Brox, Riedmiller** (2015). Striving for Simplicity: The All Convolutional Net. *ICLR Workshop*

5. **Zhou, Khosla, Lapedriza, Oliva, Torralba** (2016). Learning Deep Features for Discriminative Localization. *CVPR*

6. **Ribeiro, Singh, Guestrin** (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD*

7. **Alain, Bengio** (2016/2017). Understanding Intermediate Layers Using Linear Classifier Probes. *ICLR Workshop*

8. **Shrikumar, Greenside, Kundaje** (2017). Learning Important Features Through Propagating Activation Differences. *ICML*

9. **Sundararajan, Taly, Yan** (2017). Axiomatic Attribution for Deep Networks. *ICML*

10. **Lundberg, Lee** (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*

11. **Selvaraju, Cogswell, Das, Vedantam, Parikh, Batra** (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*

12. **Kim, Wattenberg, Gilmer, Cai, Wexler, Viegas, Sayres** (2018). Interpretability Beyond Classification Error: Applying Testing with Concept Activation Vectors (TCAV). *ICML*

13. **Adebayo, Gilmer, Muelly, Goodfellow, Hardt, Kim** (2018). Sanity Checks for Saliency Maps. *NeurIPS*

14. **Lundberg, Erion, Chen, DeGrave, Prutkin, Nair, et al.** (2020). From Local Explanations to Global Understanding with Explainable AI for Trees. *Nature Machine Intelligence*

15. **Tenney, Das, Pavlick** (2019). BERT Rediscovers the Classical NLP Pipeline. *ACL*

16. **Voita, Titov** (2020). Information-Theoretic Probing with Minimum Description Length. *EMNLP*

17. **Koh, Nguyen, Tang, Mussmann, Pierson, Kim, Liang** (2020). Concept Bottleneck Models. *ICML*

18. **Elhage, Henighan, Joseph et al.** (2022). Toy Models of Superposition. *Transformer Circuits Thread*

19. **Meng, Bau, Andonian, Belinkov** (2022). Locating and Editing Factual Associations in GPT. *NeurIPS*

20. **Wang, Variengien, Conmy, Shlegeris, Steinhardt** (2022). Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small. *ICLR 2023*

21. **Frye, de Mijolla, Begley, Cowton, Stanley, Feige** (2020). Asymmetric Shapley Values: Incorporating Causal Knowledge into Model-Agnostic Explainability. *NeurIPS 2020*

21b. **Teneggi, Bharti, Romano, Sulam** (2022). SHAP-XRT: The Shapley Value Meets Conditional Independence Testing. *arXiv:2207.07038*

22. **Goldowsky-Dill, MacLeod, Sato, Arora** (2023). Localizing Model Behavior with Path Patching. *arXiv:2304.05969*

23. **Cunningham, Ewart, Riggs, Huben, Sharkey** (2023). Sparse Autoencoders Find Highly Interpretable Features in Language Models. *ICLR 2024 (arXiv:2309.08600)*

24. **Bricken, Templeton, Batson, Chen, Jermyn, Conerly, et al.** (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. *Anthropic*

25. **Bills, Cammarata, Mossing, Tillman, Gao, Hernandez, et al.** (2023). Language Models Can Explain Neurons in Language Models. *OpenAI*

26. **Gao, la Tour, Tillman, Goh, Tow, Bahri, et al.** (2024). Scaling and Evaluating Sparse Autoencoders. *OpenAI*

27. **Tempel, Groos, Ihlen, Adde, Strümke** (2024). Choose Your Explanation: A Comparison of SHAP and Grad-CAM in Human Activity Recognition. *arXiv:2412.16003*

27b. **Nohara, Matsumoto, Soejima, Nakashima** (2022). Explanation of machine learning models using shapley additive explanation and application for real data in hospital. *Computer Methods and Programs in Biomedicine*

28. **Lindsey, Gurnee, Conmy, Adly, Zimmerman, Price, et al.** (2025). On the Biology of a Large Language Model. *Anthropic*

