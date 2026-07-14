---
author: Mehran Karimzadeh
date: July 14, 2026
description: A historical review of how AI models started to reason, from the Transformer and chain-of-thought prompting to reinforcement learning with verifiable rewards and recursive latent reasoning.
---

# How AI Learned to Think: A Historical Review

**From Transformers to Recursive Models (2017–2026)**

---

## Contents

1. [Why "Reasoning" Became the Frontier](#why-reasoning-became-the-frontier)
2. [The Four Conceptual Shifts](#the-four-conceptual-shifts)
3. [Historical Timeline](#historical-timeline)
4. [Benchmark Datasets That Drove the Field](#benchmark-datasets-that-drove-the-field)
5. **Era 1 — Foundations: The Transformer & Self-Supervised Pretraining (2017–2019)**
   - [The Transformer](#the-transformer)
   - [GPT-1, BERT, T5: Two Recipes for "Knowing"](#gpt-1-bert-t5-two-recipes-for-knowing)
   - [Universal Transformer: The Seed of Recursion](#universal-transformer-the-seed-of-recursion)
6. **Era 2 — Scale, Few-Shot, and the First Reasoning Tricks (2020–2022)**
   - [GPT-3 and In-Context Learning](#gpt-3-and-in-context-learning)
   - [Qwen-1: A Major Chinese Open Foundation Model](#qwen-1-a-major-chinese-open-foundation-model)
   - [Chain-of-Thought Prompting](#chain-of-thought-prompting)
   - [InstructGPT and ChatGPT (GPT-3.5)](#instructgpt-and-chatgpt-gpt-35)
7. **Era 3 — Alignment by Reinforcement Learning (2022–2024)**
   - [A Primer on Reinforcement Learning for Language Models](#a-primer-on-reinforcement-learning-for-language-models)
   - [RLHF and PPO in Practice](#rlhf-and-ppo-in-practice)
   - [DPO: Direct Preference Optimization](#dpo-direct-preference-optimization)
   - [GPT-4 and Tool-Augmented Reasoning](#gpt-4-and-tool-augmented-reasoning)
   - [Tree of Thoughts & Search-Based Reasoning](#tree-of-thoughts--search-based-reasoning)
8. **Era 4 — Reasoning by Reinforcement Learning with Verifiable Rewards (2024–2025)**
   - [OpenAI o1: Hidden Chain-of-Thought as a Trained Policy](#openai-o1-hidden-chain-of-thought-as-a-trained-policy)
   - [DeepSeek-R1 and GRPO: Open Reasoning at Frontier Scale](#deepseek-r1-and-grpo-open-reasoning-at-frontier-scale)
   - [Qwen3: Open Hybrid Reasoning](#qwen3-open-hybrid-reasoning)
   - [GPT-5: Routing Between Fast and Slow](#gpt-5-routing-between-fast-and-slow)
9. **Era 5 — Latent & Recursive Reasoning (2023–2026)**
   - [Looped Transformers](#looped-transformers)
   - [Coconut: Reasoning in Continuous Latent Space](#coconut-reasoning-in-continuous-latent-space)
   - [HRM: Hierarchical Reasoning Model](#hrm-hierarchical-reasoning-model)
   - [TRM: Less is More with Tiny Recursive Networks](#trm-less-is-more-with-tiny-recursive-networks)
   - [GRAM: Generative Recursive Reasoning Models](#gram-generative-recursive-reasoning-models)
10. [Comparison Matrix](#comparison-matrix)
11. [Open Challenges](#open-challenges)
12. [Summary: The Evolution of Reasoning Research](#summary-the-evolution-of-reasoning-research)
13. [References](#references)

---

## Why "Reasoning" Became the Frontier


For most of deep learning's history, a model's primary job was to recognize: to classify an image, or to translate a sentence. Reasoning, in the sense of inference that takes multiple steps, respects constraints, and can entertain counterfactuals, was long considered a niche concern of symbolic AI, separated from the neural mainstream by a deep methodological wall.

That wall fell quickly between 2020 and 2026. This period can be understood as two distinct waves of large language model (LLM) development:

### The First Wave: The Transformer Revolution
The initial breakthrough was driven by transformer-based architectures and defined by two key training paradigms: massive pretraining on web text and fine-tuning on human feedback. This first wave gave models unprecedented fluency and pattern recognition capabilities. These advances rapidly transformed general computing and successfully translated to specialized domains, significantly impacting fields like biomedical AI.

### The Second Wave: The Reasoning Revolution
In the next wave, models acquire abilities that behaviorally resemble multi-step thought. This era is driven by a third training paradigm, reinforcement learning on problems whose answers can be checked automatically, and it tackles reasoning head-on. Reasoning is the operational definition of intelligence behind benchmarks such as Sudoku, the Abstraction and Reasoning Corpus (ARC-AGI), mathematics olympiads, and scientific problem-solving.

This second wave of reasoning is being operationalized through two distinct approaches:

* The sequence-extension approach: models like GPT, Qwen, and OpenAI's o-series generate increasingly long reasoning traces, supervised by either human preferences or task-grounded rewards.

* The recursion approach: more recent architectures, such as the Universal Transformer, the Hierarchical Reasoning Model (HRM), the Tiny Recursive Model (TRM), and the Generative Recursive reAsoning Model (GRAM), refine a fixed-size internal state by applying the same small network over and over. This decouples reasoning depth from both parameter count and output length. While these models have not passed the test of time, many are excited by the innovative contributions which bring clear evidence of the densing law to reasoning literature.

### The Path Forward

While the advances in data engineering and model architecture from the first wave of LLMs have already reshaped biomedical AI, the complex capabilities of the second wave have yet to be fully translated. Understanding the key components of these new reasoning and recursive models is particularly crucial for biomedical AI scientists seeking to integrate true multi-step inference into the next generation of scientific applications.

## The Four Conceptual Shifts

Looking back across the decade, four independent shifts organize everything. A model might adopt one shift and not the others, but the frontier models combine all four.

**1. Training data: from raw text to instructions, preferences, and verifiable problems**

- **Raw text** (2018–2022): unfiltered internet (Common Crawl, books, code). Objective: next-token likelihood.

- **Instruction / response pairs** (2022–): curated demonstrations. Often called supervised fine-tuning (SFT).

- **Preference pairs** (2022–): two answers to the same prompt, one marked as preferred by a human. Drives reward modeling and reinforcement learning from human feedback (RLHF).

- **Verifiable problems with a checker** (2024–): math, code, and formal logic, where answers can be graded automatically. Drives reinforcement learning with verifiable rewards (RLVR).

- **Self-generated trajectories under a checker** (2024–): the model proposes its own training data; only correct trajectories receive positive gradient.

**2. Objective: from likelihood to reward to a bound on latent trajectories**

- **Next-token cross-entropy**: the pretraining objective. The model maximizes $\mathbb{E}\, \log p_\theta(x_{t+1} | x_{\le t})$, the average log-probability it assigns to each token $x_{t+1}$ given the preceding tokens $x_{\le t}$, where $\theta$ denotes the model parameters.

- **Preference (Bradley–Terry) likelihood**: used to fit reward models from human choices between two answers.

- **Expected reward with a Kullback–Leibler (KL) penalty** that keeps the model near a reference: the territory of Proximal Policy Optimization (PPO), Group Relative Policy Optimization (GRPO), and Direct Preference Optimization (DPO), all unpacked in Era 3 and 4.

- **Verifiable scalar reward**: a checker assigns each output $y$ a reward $R(y)$ of 1 if it is correct and 0 otherwise (a math answer matches, unit tests pass).

- **Evidence lower bound (ELBO)** over latent reasoning trajectories: GRAM's variational view of recursion, explained in Era 5.

**3. Architecture: from fixed-depth Transformer to recurrent reasoning blocks**

- **Stacked Transformer blocks** (each layer used once): GPT, BERT, T5.

- **Mixture of experts (MoE)**: only a fraction of the network activates for each token, so the model can hold more parameters at constant compute (Qwen-MoE, DeepSeek-V3).

- **Looped / Universal Transformers**: a single block applied many times, reusing the same weights at every step.

- **Hierarchical recurrence**: two modules updating at different time scales (HRM's high-level and low-level states).

- **Tiny recursive networks**: 7 million parameters refining a state for hundreds of steps (TRM).

- **Stochastic latent transitions**: a Gaussian noise term steers reasoning down branching paths (GRAM).

**4. Inference: from one forward pass to test-time computation**

- **Single forward pass**: original GPT, BERT.

- **Longer outputs as more compute**: chain-of-thought prompting, then o1 and R1.

- **Sampling plus voting or verification**: generate many answers, keep the most common or the highest-scored one (self-consistency, best-of-N selection, process reward models).

- **Recursive depth**: more iterations of a shared block, at no extra parameters (HRM, TRM).

- **Parallel latent trajectories**: sample many independent reasoning paths and aggregate them, scaling in width rather than depth (GRAM).

## Historical Timeline

- **2017**: Transformer (Vaswani et al.)

- **2018**: GPT-1; BERT; Universal Transformer (Dehghani et al.)

- **2019**: GPT-2; T5; RoBERTa

- **2020**: GPT-3 (in-context learning); scaling laws

- **2022 Jan**: Chain-of-Thought Prompting (Wei et al.)

- **2022 Mar**: InstructGPT, RLHF at scale (Ouyang et al.)

- **2022 Nov**: ChatGPT (GPT-3.5) public release

- **2023 Mar**: GPT-4; LLaMA-1

- **2023 Apr**: Qwen-1 announced (Alibaba); weights opened in August

- **2023 May**: Tree of Thoughts (Yao et al.); DPO (Rafailov et al.)

- **2023 Nov**: Looped Transformers (Yang et al.)

- **2024 Sep**: OpenAI o1, hidden chain-of-thought policy

- **2024 Dec**: Coconut, Chain-of-Continuous-Thought (Hao et al.)

- **2025 Jan**: DeepSeek-R1 and GRPO, open reasoning at the frontier

- **2025 Apr**: OpenAI o3; ARC-AGI-2 (Chollet et al.)

- **2025 May**: Qwen3, hybrid thinking/non-thinking modes

- **2025 Jun**: HRM, Hierarchical Reasoning Model (Wang et al.)

- **2025 Aug**: GPT-5, router between fast and reasoning modes

- **2025 Oct**: TRM, Tiny Recursive Model (Jolicoeur-Martineau)

- **2026 May**: GRAM, Generative Recursive Reasoning (Baek et al.)

## Benchmark Datasets That Drove the Field

Reasoning research has moved through several waves of benchmarks, each one designed to be hard for the previous generation's models. Knowing what these datasets contain, and how they were built, is important context for the per-model results quoted throughout this review.

#### Math word problems and competition math

**GSM8K (Grade School Math 8K)**

Released by OpenAI (Cobbe et al., 2021). Grade-school math word problems written by human contractors, split into 7,473 training and 1,319 test examples (8,792 in total, colloquially "8.5K"). Each problem has a natural-language statement, a step-by-step solution, and a single numeric answer. The dataset became the defining test of arithmetic reasoning for a generation of language models, and it was the benchmark on which Chain-of-Thought prompting first showed dramatic gains (Wei et al., 2022, lifting PaLM-540B from roughly 18 percent to 57 percent).

**MATH**

Released by Hendrycks et al. (2021). Twelve thousand five hundred problems drawn from American high-school mathematics competitions (AMC 10, AMC 12, AIME), categorized across seven subjects (algebra, geometry, number theory, counting and probability, intermediate algebra, prealgebra, precalculus) and five difficulty levels. Seven thousand five hundred training examples and five thousand test examples. Solutions are written in LaTeX, and final answers must match exactly, which makes MATH amenable to automatic verification.

**AIME (American Invitational Mathematics Examination)**

A real annual competition for top US high school students. Each contest is a three-hour test with fifteen problems whose answers are integers between 0 and 999, which makes scoring trivially automatic. Because the test is reissued every year, contamination of pretraining corpora is limited, so AIME-2024 and AIME-2025 became favorite evaluations for frontier reasoning models. OpenAI o1 scored 83 percent on AIME-2024 versus GPT-4o's 13 percent, a result that put RL-trained reasoning on the map.

#### Code generation

**HumanEval**

Released by Chen et al. (2021) alongside Codex. One hundred sixty-four hand-written Python programming problems, each consisting of a function signature, a docstring describing the desired behavior, and a hidden set of unit tests. A solution counts as correct only if it passes every test. The verifier is the Python interpreter, which makes HumanEval one of the earliest benchmarks suited to reinforcement learning with verifiable rewards. The pass@k metric (probability that at least one of k samples passes) was introduced here.

**Codeforces and IOI**

Live competitive programming. Codeforces ratings give a continuous skill ladder (an Elo-style rating, as in chess); o1 reached a rating of about 1807, placing it in the eighty-ninth percentile of human competitors. The International Olympiad in Informatics 2024 (six problems, hidden test suites, very long time limits) became an explicit target for o1 and DeepSeek-R1, both of which reached gold-medal territory under sufficient inference budgets.

#### General reasoning and knowledge

**MMLU (Massive Multitask Language Understanding)**

Released by Hendrycks et al. (2020). Fifteen thousand nine hundred eight multiple-choice questions spanning fifty-seven subjects, from elementary mathematics to professional law and clinical medicine. Each question has four options and one correct answer. MMLU is the de facto knowledge breadth benchmark; in 2020 GPT-3 reached 43.9 percent (random is 25 percent), in 2023 GPT-4 reached 86.4 percent, and frontier models in 2025 routinely exceed 90 percent.

**BIG-Bench and BIG-Bench Hard**

A collaborative benchmark of more than two hundred tasks, contributed by 442 authors and released by Srivastava et al. (2022). Tasks range from logical deduction to social bias to mathematical induction. BIG-Bench Hard (Suzgun et al., 2022) extracted the twenty-three subtasks where GPT-3-era models still underperformed the average human rater, and this subset became a standard reasoning probe.

#### Abstract and structured reasoning, the focus of the recursive-reasoning line

**ARC-AGI-1**

Introduced by François Chollet (2019) in his paper "On the Measure of Intelligence" and used since 2020 as the ARC Prize benchmark. The full set contains roughly 1,000 visual reasoning tasks split into 400 training, 400 public evaluation, and 200 private evaluation tasks. Each task gives between two and six demonstration pairs of input grid and output grid, and the model must produce the correct output for a held-out test input. Grids are 1×1 to 30×30 colored cells. ARC-AGI was explicitly designed to resist memorization and statistical pattern matching, so progress here was slow until OpenAI o3 hit 87.5 percent in late 2024 at very high inference cost.

**ARC-AGI-2**

Released by Chollet et al. (2025). A harder successor that keeps the same input/output grid format but designs each task to require multi-step transformation and to be unsolvable by pure pattern matching. Frontier reasoning LLMs scored in the single digits on ARC-AGI-2 at launch (GPT-5.2 low at 9.7 percent, Grok-4 thinking at 16.0 percent), and TRM and GRAM operate in the 3 to 11 percent range despite using a tiny fraction of the parameters.

**Sudoku-Extreme**

Introduced by Wang et al. (2025) alongside HRM. Standard 9×9 Sudoku boards filtered for very low clue counts (typically seventeen, the proven minimum for a unique solution). The dataset contains tens of thousands of such puzzles. Sudoku-Extreme is solvable by deterministic constraint propagation plus search, but reaching high accuracy with a feedforward model requires extended computation: HRM scores 55 percent, TRM 87 percent, GRAM 97 percent at the same model scale.

**N-Queens and Graph Coloring**

Classical constraint satisfaction problems used by Baek et al. (2026) to evaluate multi-solution behavior. N-Queens asks for a placement of N non-attacking queens on an N×N board; the 8×8 case has 92 distinct solutions, the 10×10 case has 724. Graph Coloring asks for a vertex coloring using at most K colors such that no two adjacent vertices share a color. These problems are critical to the GRAM paper because they have many valid solutions per instance, exposing the mode collapse of deterministic recursive models.

#### Generation, used to probe unconditional modeling


**Inception Score (IS) and Fréchet Inception Distance (FID)**

Standard metrics for image generation, not datasets themselves but worth knowing. The Inception Score (Salimans et al., 2016) needs no reference images: it scores generated images using an image classifier pretrained on ImageNet, rewarding samples that are individually confident (the classifier assigns each image a clear class) and collectively diverse (the predicted classes vary across the sample set). Higher is better. The Fréchet Inception Distance (Heusel et al., 2017), by contrast, requires a set of real reference images: it summarizes the classifier's internal features of generated and real samples as two Gaussian distributions and measures the distance between them. Lower is better.

> **Insight:** Note the trajectory of difficulty. GSM8K (grade-school math) is largely saturated by frontier models, MATH and HumanEval are largely solved or close to it, AIME and Codeforces remain meaningful at frontier scale, and ARC-AGI-2 plus the structured reasoning suites used by HRM, TRM, and GRAM are where the current research frontier lives. A useful benchmark in 2026 is one that resists scale and forces models to actually think.

## Era 1 — Foundations: The Transformer & Self-Supervised Pretraining (2017–2019)

*A new architecture meets a simple objective. Reasoning is not yet a goal, but the machinery is being assembled.*

### The Transformer

**Vaswani et al., 2017** | *Architecture · Attention*
Recurrent neural networks, such as the long short-term memory (LSTM) network, process a sequence one step at a time. That serial bottleneck limits both training parallelism and the ability to connect distant parts of a sequence. The Transformer dispenses with recurrence entirely. Its core operation is **scaled dot-product attention**, in which every token in a sequence directly attends to every other:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

*Each token contributes a query vector, a key vector, and a value vector; stacking them across tokens gives the matrices $Q$, $K$, and $V$. The product $QK^\top$ scores how relevant every token is to every other. Dividing by $\sqrt{d_k}$, where $d_k$ is the dimension of the key vectors, keeps those scores in a numerically stable range as dimensionality grows. The softmax then turns each row of scores into a probability distribution, and those probabilities are used to take a weighted average of the value vectors.*

Multiple attention "heads" run this operation in parallel; their outputs are concatenated and projected back to the model dimension. Each of the model's stacked Transformer blocks contains a multi-head self-attention layer followed by a position-wise feed-forward network (a small multi-layer perceptron applied identically at each position), with residual connections and layer normalization around each. Crucially, all of this is **fully parallel across sequence positions** at training time, unlocking GPU throughput that recurrent networks could not match.

> **Insight:** The Transformer's enduring assumption is depth-as-power: more reasoning means more stacked blocks. Each block is used exactly once per forward pass. This is precisely the assumption that Universal Transformers, HRM, TRM, and GRAM will later challenge.

### GPT-1, BERT, T5: Two Recipes for "Knowing"

**2018–2019** | *Pretraining · Self-Supervised*
The Transformer is just an architecture; what made it transformative was the discovery that *self-supervised pretraining on enormous text corpora* produces representations that transfer to almost every downstream natural language processing task. Three flavors emerged:

**GPT-1 (2018): decoder-only**

GPT stands for Generative Pre-trained Transformer. 117 million parameters, trained by causal (left-to-right) language modeling: predict the next token from all previous tokens. Suited to generation. This is the lineage that will become ChatGPT.

**BERT (2018): encoder-only**

BERT stands for Bidirectional Encoder Representations from Transformers. 340 million parameters, trained by *masked* language modeling: hide 15% of tokens and predict them from the surrounding context on both sides. Suited to classification, retrieval, and natural language inference. A different lineage, not directly used for generating reasoning.

**T5 (2019): encoder-decoder**

The "Text-to-Text Transfer Transformer" frames every language task as text in, text out. A forerunner of the unified instruction-tuned approach that will dominate from 2022.

$$
\mathcal{L}_{\text{GPT}}(\theta) = -\sum_{t=1}^{|x|} \log p_\theta(x_t \mid x_{<t})
$$

*The next-token cross-entropy loss. For every position $t$ in a training text $x$, the model (with parameters $\theta$) is pushed to assign high probability $p_\theta$ to the actual next token $x_t$ given all the tokens before it, written $x_{<t}$; the sum runs over the whole text of length $|x|$. The entire reasoning capability of GPT-3, and even the base model behind GPT-4, emerges from this single objective applied to enough text.*

#### Data shift: from labelled to unlabelled

The decisive change here was the willingness to throw away labels altogether. GPT-1's pretraining corpus (BookCorpus) had no human annotations; it was simply text. The model learned grammar, world facts, basic arithmetic, and rudimentary reasoning *as a side effect* of predicting what word comes next. Subsequent generations would scale this corpus by orders of magnitude, and reasoning would scale with it.

### Universal Transformer: The Seed of Recursion

**Dehghani et al., 2018** | *Recursive · ACT*
While the mainstream stacked Transformer blocks deeper, Dehghani and colleagues at Google asked a different question: what if instead of stacking different blocks, we apply *one block* over and over? Each position passes through the same Transformer layer for *T* steps, with optional **Adaptive Computation Time (ACT)**, a learned mechanism to decide, per position, how many steps to compute before halting.

$$
h^{(t)}_i = \text{TransformerBlock}\!\left(h^{(t-1)}; \theta\right) + h^{(t-1)},\quad t=1,\dots,T
$$

*Weight-sharing across depth: $h^{(t)}_i$ is the hidden representation of sequence position $i$ after $t$ applications of the block, and one set of parameters $\theta$ is reused at every step from $1$ to $T$. Adding the previous state back in (the trailing $+\,h^{(t-1)}$) is a residual connection that stabilizes training. Reasoning depth is decoupled from parameter count.*

The Universal Transformer was theoretically appealing (it could simulate algorithms of variable runtime), but on natural-language benchmarks it was outperformed by larger stacked models, and the field largely moved on. The idea would lie dormant until 2023, when **Looped Transformers**, then **HRM**, **TRM**, and finally **GRAM** revived it as the central mechanism of recursive reasoning models.

## Era 2 — Scale, Few-Shot, and the First Reasoning Tricks (2020–2022)

*Reasoning emerges not from new objectives but from sheer scale, plus a single magical phrase: "Let's think step by step."*

### GPT-3 and In-Context Learning

**Brown et al., 2020** | *Scale · Few-Shot*
GPT-3 scaled the GPT recipe to 175 billion parameters and trained on roughly 300 billion tokens of filtered Common Crawl, WebText, books, and Wikipedia. The architecture was essentially unchanged from GPT-2 (a decoder-only Transformer, with sparse attention patterns added), and the objective was still pure next-token cross-entropy.

The headline finding was an emergent ability: at this scale the model could perform new tasks from a handful of examples placed in the prompt, with no gradient updates at all. This was named **in-context learning**. Translation, arithmetic, simple word algebra, and analogy completion all became accessible by simply *showing* the model a few input-output examples and a fresh query.

> **Insight:** In-context learning hinted that reasoning might already be latent in a well-trained language model; perhaps no special architecture was required, just the right prompt. This intuition would shape the next two years and lead directly to chain-of-thought prompting.

#### Component note: sparse attention patterns

Attention cost grows with the square of sequence length, so GPT-3 alternated dense attention layers with sparse ones in which each token attends only to a local band of neighbors. Subsequent open models (Qwen, LLaMA) returned to fully dense attention and instead managed long contexts with rotary position embeddings (covered below), grouped-query attention, and faster attention implementations such as FlashAttention.

### Qwen-1: A Major Chinese Open Foundation Model

**Alibaba, April 2023 (announced); Qwen-7B open weights August 2023** | *Open Weights · Multilingual*
Qwen was Alibaba's response to GPT-3 and LLaMA: a family of decoder-only Transformers pretrained on roughly 2.4 trillion tokens of multilingual text with a strong Chinese component, including code and mathematical text. **Qwen-7B and Qwen-14B** were open-sourced later that year. 

**Architectural choices in Qwen-1**

- **Rotary position embeddings (RoPE)**, which have since become the industry standard way to encode token positions, and which behave gracefully even at long context lengths.

- **SwiGLU activation** in the feed-forward block, empirically superior to the more common GELU activation (both are unpacked below).

- **RMSNorm** in place of LayerNorm: slightly cheaper to compute, with comparable quality.

- **Untied input and output embeddings**, and a large 152K-entry vocabulary built with byte pair encoding to represent Chinese characters efficiently. Byte pair encoding builds a vocabulary of common subwords by repeatedly merging the most frequent pair of adjacent symbols in the training text.

Each of those three component swaps (RoPE, SwiGLU, RMSNorm) is small in isolation, but together they define what is now the default modern decoder-only stack, used by LLaMA, Mistral, Qwen, DeepSeek, and Gemma. They have become standard in many biomedical foundation models as well, so each is worth understanding on its own terms.

#### SwiGLU activation: why it beats GELU

Within the standard Transformer feed-forward block, two linear layers have a pointwise nonlinearity in between:

$$
\mathrm{FFN}_{\mathrm{GELU}}(x) \;=\; \mathrm{GELU}(x W_1)\, W_2,
      \qquad
      \mathrm{GELU}(x) \;=\; x \cdot \Phi(x)
$$

*$x$ is the input activation vector, $W_1$ and $W_2$ are the block's two learned weight matrices, and $\Phi$ is the cumulative distribution function of the standard Gaussian. GELU (the Gaussian Error Linear Unit) is a smooth, slightly non-monotonic alternative to the standard ReLU activation.*

SwiGLU (Shazeer, 2020) replaces this with a *gated* linear unit. Two parallel projections of $x$ are computed; one passes through a Swish activation (also called SiLU, the sigmoid linear unit), the other is left linear, and the two are multiplied elementwise before the final output projection:

$$
\mathrm{SwiGLU}(x) \;=\; \bigl(\, \mathrm{Swish}(x W_1) \,\odot\, x W_3\,\bigr)\, W_2,
      \qquad
      \mathrm{Swish}(z) \;=\; z \cdot \sigma(z)
$$

*Three weight matrices instead of two. $W_1$ feeds the gate (after Swish), $W_3$ feeds the value, and $W_2$ is the output projection. The symbol $\odot$ denotes elementwise multiplication, and $\sigma$ is the logistic sigmoid $\sigma(z) = 1/(1+e^{-z})$.*

Why does this empirically beat GELU? Three reasons:

- **Multiplicative interactions.** GELU applies a single pointwise function. SwiGLU multiplies two input-dependent quantities together, which makes the feed-forward block strictly more expressive: this *data-dependent gating* can suppress or amplify each feature based on the current activation, not just reshape it.

- **Smooth, non-monotonic Swish.** Like GELU, Swish has a small dip just below zero that empirically improves optimization. Unlike GELU, it interacts multiplicatively with another learned projection, which seems to matter more than the precise shape of the activation.

- **Parameter-matched comparisons still favor SwiGLU.** The three-matrix block looks more expensive, but practitioners shrink the hidden dimension to two-thirds of its usual size so that total parameters and floating-point operations match a two-matrix GELU block. Even at matched cost, Shazeer reported consistent perplexity improvements on the T5 pretraining task, with similar gains later confirmed at LLaMA scale.

#### Rotary position embeddings (RoPE): why they extrapolate

The original Transformer added either sinusoidal or learned *absolute* position weights to the input embeddings. Two problems followed. First, an absolute encoding scheme has no obvious way to score positions beyond the training length. Second, position and content compete for the same vector space because they are added together at the input layer.

RoPE (Su et al., 2021) fixes both by injecting position information directly into the attention dot product as a *rotation*. For a query or key vector $q \in \mathbb{R}^d$, split the dimensions into pairs and rotate each pair by an angle proportional to position $m$:

$$
R_{\Theta, m} \,q \;=\;
      \begin{pmatrix}
        \cos m\theta_1 & -\sin m\theta_1 & & & \\
        \sin m\theta_1 &  \cos m\theta_1 & & & \\
                       &                 & \cos m\theta_2 & -\sin m\theta_2 & \\
                       &                 & \sin m\theta_2 &  \cos m\theta_2 & \\
                       &                 &                &                 & \ddots
      \end{pmatrix}
      \begin{pmatrix} q_1 \\ q_2 \\ q_3 \\ q_4 \\ \vdots \end{pmatrix}
$$

*A block-diagonal rotation matrix acting on the query vector $q = (q_1, q_2, q_3, q_4, \dots)$. Each two-dimensional sub-block, made of consecutive coordinate pairs like $(q_1, q_2)$, is rotated by the angle $m\theta_i$, where $m$ is the token's position and the frequencies $\theta_i = 10000^{-2(i-1)/d}$ for $i = 1, \dots, d/2$ range from fast-rotating to slow-rotating. Key vectors receive the same rotation at their own positions.*

The decisive property: when the rotated query at position $m$ is dotted with the rotated key at position $n$, the result depends only on the *difference* $m - n$:

$$
\langle R_{\Theta, m}\, q,\; R_{\Theta, n}\, k \rangle
      \;=\; \langle q,\; R_{\Theta, n-m}\, k \rangle.
$$

*The angle brackets $\langle \cdot, \cdot \rangle$ denote the dot product, the operation at the heart of the attention score. Attention scores thus become a function of relative position by construction, with no learned per-position parameters.*

The practical consequences include:

- **Length extrapolation.** Because no parameters are tied to specific positions, the model can be queried at positions beyond the training context, with graceful degradation. Follow-up techniques such as position interpolation (Chen et al., 2023) and YaRN (Peng et al., 2023) stretch RoPE-trained models from a few thousand to hundreds of thousands of tokens of context without retraining from scratch.

- **Position lives in the attention layer.** RoPE rotates queries and keys only; the value vectors and the residual stream carry pure content. Position and meaning travel through separate pathways.

- **Translation-equivariant.** Two identical phrases at different positions produce the same relative-attention structure, which matches the semantic intuition behind attention.

- **Cheap.** A rotation is two multiplications per pair of dimensions; it costs almost nothing compared to the attention matrix product itself, and there are no learned position parameters.

#### RMSNorm: why dropping the mean helped

The standard Transformer normalizes activations using LayerNorm. For an activation vector $x \in \mathbb{R}^d$:

$$
\mathrm{LayerNorm}(x) \;=\; \gamma \,\odot\, \frac{x - \mu(x)}{\sqrt{\sigma^2(x) + \varepsilon}} \;+\; \beta,
      \qquad
      \mu(x) = \tfrac{1}{d}\sum_i x_i, \quad
      \sigma^2(x) = \tfrac{1}{d}\sum_i (x_i - \mu)^2.
$$

*LayerNorm subtracts the vector's mean $\mu(x)$, divides by its standard deviation (the square root of the variance $\sigma^2(x)$, with a small constant $\varepsilon$ to avoid division by zero), and then applies learned per-dimension scale and shift parameters $\gamma$ and $\beta$. Two passes over the vector: one to center, one to rescale.*

RMSNorm (Zhang and Sennrich, 2019) keeps only the rescaling. It divides by the root-mean-square of the activations and drops the mean subtraction entirely:

$$
\mathrm{RMSNorm}(x) \;=\; \gamma \,\odot\, \frac{x}{\mathrm{RMS}(x)},
      \qquad
      \mathrm{RMS}(x) = \sqrt{\tfrac{1}{d}\sum_i x_i^2 + \varepsilon}.
$$

*No mean computation and no shift parameter $\beta$; the vector is simply divided by its root-mean-square and rescaled by $\gamma$. One pass over the vector instead of two.*

Zhang and Sennrich's central observation was that the stabilizing effect of LayerNorm comes almost entirely from rescaling, not from recentering. Empirically, dropping the mean term costs nothing on Transformer pretraining quality, and the improvements include:

- **About 7 to 64 percent faster** in their original benchmarks on recurrent and Transformer architectures, because the mean computation and subtraction are eliminated and only one reduction over the $d$ dimensions is needed instead of two.

- **Numerically friendlier in low precision.** The mean-subtraction step in LayerNorm can produce cancellation errors in the 16-bit floating-point formats used for large-scale training; RMSNorm avoids that path entirely. This matters at the scale Qwen, LLaMA, and Mistral train at.

- **Fewer parameters.** The $\beta$ bias is gone; modern implementations often initialize $\gamma$ to a constant near one and learn very little, which simplifies optimization.

- **Adopted by the modern open-weights stack.** LLaMA, PaLM, Mistral, DeepSeek, Gemma, and Qwen all use RMSNorm in pre-norm position (normalization applied *before* the residual branch, not after). The combination of pre-norm + RMSNorm is now standard for billion-scale and trillion-scale training stability.

> **Insight:** Together, RoPE plus SwiGLU plus RMSNorm illustrate a broader pattern in modern LLM design. Each replaces a generic 2017-era component with a slightly more specialized variant: position information moves from input embeddings to attention rotations, the feed-forward block gains multiplicative gating, and normalization drops the operation that turned out not to be load-bearing. None of these is a paradigm shift on its own, but together they save compute, improve quality, and unlock longer contexts. Qwen-1 was an early adopter of all three; Qwen3 still uses the same trio.

Qwen-1's pretraining data already contained substantial code and math. On top of that, a supervised fine-tuning phase used about 500K instruction-response pairs, followed by an RLHF round (PPO against a reward model trained on human preferences; all of this machinery is unpacked in Era 3). That made Qwen-1 one of the first open models to bundle the full pretrain, fine-tune, RLHF pipeline. We trace its evolution to Qwen3 in Era 4.

### Chain-of-Thought Prompting

**Wei et al., January 2022** | *Prompting · Test-Time*
The simplest, cheapest, and most influential reasoning advance of the era. Wei and colleagues showed that prompting a sufficiently large language model with *worked examples that include intermediate steps* dramatically improves performance on math word problems, symbolic reasoning, and commonsense tasks.

The trigger was almost embarrassingly small: prepend "Let's think step by step" to a prompt, or include an exemplar of the form "*Q: ... A: First, ... Therefore the answer is ...*". On GSM8K (grade-school math), this lifted PaLM-540B from 18% to 57%.

**Why does it work?**

- Generation is autoregressive. Each new token uses all preceding tokens as context, so writing "the sum is 14, half of which is 7" gives the next step direct access to the intermediate result instead of forcing the model to compute it internally in a single forward pass.

- The Transformer has a fixed depth; arbitrarily long chains of reasoning would otherwise be limited by the number of layers. Writing intermediate tokens lets reasoning depth grow with output length.

- Pretraining corpora contain many worked examples (textbooks, Stack Exchange, GitHub). The model has seen the pattern.

Chain-of-thought prompting was followed quickly by *self-consistency* (Wang et al., 2022), which samples many chains and picks the majority answer, and later by *Tree of Thoughts*, *Graph of Thoughts*, and program-augmented variants. All of these are inference-time techniques: no retraining, just clever decoding.

### InstructGPT and ChatGPT (GPT-3.5)

**Ouyang et al., March 2022 → ChatGPT release November 2022** | *SFT · RLHF*
GPT-3 was knowledgeable but not helpful: prompted with a question, it was as likely to autocomplete with a list of similar questions as to answer one. InstructGPT introduced a three-stage pipeline that would become the default recipe for every commercial chat model:

**The InstructGPT pipeline**

- **Stage 1, Supervised Fine-Tuning (SFT).** Human labelers wrote about 13K demonstrations of how a helpful assistant should respond to prompts. Standard cross-entropy fine-tuning on these demonstrations turns a raw language model into an instruction-follower.

- **Stage 2, reward model training.** For each prompt, the fine-tuned model produced several candidate responses, and a human ranked them. A separate model was trained to predict these rankings; the result is a learned reward function over text.

- **Stage 3, reinforcement learning from human feedback (RLHF).** The fine-tuned model was further trained with the Proximal Policy Optimization (PPO) algorithm to maximize the reward model's score, with a penalty term pulling it back toward the supervised starting point. Era 3 below explains this machinery in detail.

**ChatGPT**, launched 30 November 2022, was a sibling model to InstructGPT fine-tuned with the same SFT + RLHF recipe and wrapped in a chat interface. OpenAI later exposed it through the API as `gpt-3.5-turbo` (the related completion model `text-davinci-003` was a separate product). It became the fastest-growing consumer product in history (100M users in two months), turning what had been a research artifact into the public's first contact with modern AI.

> **Insight:** The conceptual breakthrough of ChatGPT was not new architecture but a new *objective*: optimize expected human preference (via a reward model) rather than next-token likelihood on the open web. The model now had a teacher with opinions about quality, not just consistency.

We unpack the mechanics of RLHF in the reinforcement learning primer that opens the next era.

## Era 3 — Alignment by Reinforcement Learning (2022–2024)

*The community accepts that supervised data is not enough. Reward signals, human or learned, become a core training tool.*

### A Primer on Reinforcement Learning for Language Models

**Background, Sutton & Barto 1998** | *RL Foundations · Policy Gradients*
Most readers know language models as supervised predictors: show the model an input, grade its output against a known target. Reinforcement learning (RL) works differently: the model acts, receives a scalar score for how well it did, and adjusts itself to score higher next time. To follow what InstructGPT, DPO, GRPO, R1, o1, and GRAM actually do, let's review the key concepts.

**The core objects of reinforcement learning**

- **State $s$**, what the agent observes. For a chat model, the state is the prompt plus everything the model has written so far.

- **Action $a$**, what the agent does. For a chat model, the action is the next token. Many tokens together form a *trajectory* $\tau = (s_0, a_0, s_1, a_1, \dots, s_T)$.

- **Policy $\pi_\theta(a \mid s)$**, the agent itself, parameterized by $\theta$. For a language model, $\pi_\theta$ is the next-token distribution at every state.

- **Reward $r(s, a)$ or terminal reward $R(\tau)$**, a scalar signal of how well the agent is doing. For a chat model, the reward might come from a learned reward model, a unit-test pass/fail, or a math-answer checker.

- **Return $G_t = \sum_{k \ge t}\gamma^{k-t}\, r_k$**, the discounted sum of future rewards from time $t$ onward. The discount factor $\gamma \in [0, 1)$ shrinks the weight of rewards the further in the future they arrive.

- **Value functions**, $V^\pi(s)$ is the expected return when following $\pi$ from state $s$; $Q^\pi(s,a)$ is the expected return after taking action $a$ in state $s$ and then following $\pi$.

- **Advantage $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$**, how much better action $a$ is at state $s$ than the policy's average behavior. Used to weight policy updates.

#### The objective

Reinforcement learning maximizes the expected return of trajectories drawn from the policy:

$$
J(\theta) \;=\; \mathbb{E}_{\tau \sim \pi_\theta}\!\bigl[\,R(\tau)\,\bigr]
$$

*We want to find policy parameters $\theta$ that produce trajectories with high reward. The expectation is over the distribution of trajectories the policy itself generates, which is what makes this hard.*

#### Policy gradient: the foundational identity

How do we compute $\nabla_\theta J(\theta)$ when the expectation depends on $\theta$ and $R$ might not be differentiable? The classic *REINFORCE* identity (Williams, 1992):

$$
\nabla_\theta J(\theta) \;=\; \mathbb{E}_{\tau \sim \pi_\theta}\!\Biggl[\,\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot R(\tau)\Biggr]
$$

*This is the "log-likelihood trick." The term $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ points in the parameter direction that makes action $a_t$ more likely in state $s_t$; multiplying it by the trajectory's total reward $R(\tau)$ scales that push by how well the trajectory turned out. The net effect: upweight actions from high-reward trajectories, downweight actions from low-reward ones. The variance of this estimate is enormous in practice, hence the modern modifications.*

Replacing $R(\tau)$ with the advantage $A_t$ (which reduces variance) and adding a clipping device (which preserves stability) gives **Proximal Policy Optimization (PPO)**, the workhorse of RLHF.

#### The KL leash

Pure reward maximization is dangerous. The reward model is a learned approximation; if the policy strays far from what the reward model has seen, it may discover absurd outputs that the model rates highly but humans would reject (this is *reward hacking*). RLHF therefore adds a penalty based on the Kullback–Leibler (KL) divergence, a standard measure of how different two probability distributions are:

$$
J_{\text{RLHF}}(\theta)
      = \mathbb{E}_{x,\, y \sim \pi_\theta(\cdot \mid x)}\!\bigl[\, r_\phi(x, y) \,\bigr]
        \;-\; \beta \cdot \mathrm{KL}\!\bigl(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\bigr)
$$

*Here $x$ is a prompt and $y$ is a response sampled from the current policy $\pi_\theta$. The first term is the average score the learned reward model $r_\phi$ gives those responses. The second term penalizes divergence between the policy and $\pi_{\text{ref}}$, a frozen copy of the supervised fine-tuned model, with the coefficient $\beta$ controlling how far the policy may drift. Without the KL term, RLHF collapses to gibberish that the reward model has not learned to penalize.*

> **Insight:** Three reward sources will recur in this review: **(a) learned preference models** (RLHF, used in ChatGPT, Qwen, and Claude); **(b) verifiable checkers** for math and code (used in o1, R1, and Qwen3); and **(c) learned value heads** like GRAM's Latent Process Reward Model in Era 5. The mathematics of policy optimization is identical in each case; only the source of the reward changes.

### RLHF and PPO in Practice

**Schulman et al. 2017 (PPO); Ouyang et al. 2022 (RLHF for LMs)** | *PPO · Reward Model*
PPO is the dominant algorithm for reinforcement-learning fine-tuning of language models. Its objective rewrites the policy gradient in terms of a **probability ratio** $\rho_t$ between the current policy $\pi_\theta$ and an "old" policy $\pi_{\theta_{\text{old}}}$ (a snapshot from before the current optimization step), clipped to prevent over-large updates:

$$
L^{\text{PPO}}(\theta) =
      \mathbb{E}_t\!\left[\,\min\!\Bigl(\,\rho_t(\theta)\, A_t,\;
        \mathrm{clip}\bigl(\rho_t(\theta),\,1-\epsilon,\,1+\epsilon\bigr)\, A_t\Bigr)\,\right]
      ,\qquad
      \rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

*$A_t$ is the advantage from the primer above: how much better this action was than the policy's average. The clip function caps the ratio $\rho_t$ within $[1-\epsilon,\, 1+\epsilon]$ (with $\epsilon \approx 0.2$), preventing the policy from changing too drastically in one step. Taking the minimum of the clipped and unclipped terms makes the objective a pessimistic lower bound: a policy improvement counts only if it would still look good under modest deviation.*

**Reward modeling: the Bradley–Terry detour**

How is the reward model $r_\phi$ trained? Humans rarely produce numeric scores; they pick a winner between two responses. The *Bradley–Terry* model relates pairwise preference to an underlying score:

$$
P(y_w \succ y_l \mid x) = \sigma\!\bigl(r_\phi(x, y_w) - r_\phi(x, y_l)\bigr)
$$

*$y_w$ is the response the human preferred (the winner) and $y_l$ the rejected one (the loser); the symbol $\succ$ reads "is preferred to." $\sigma$ is the logistic sigmoid, so the further the reward model $r_\phi$ scores the winner above the loser, the closer the predicted preference probability gets to one.*

Maximum-likelihood training under this model yields a reward function that predicts which of two responses humans prefer, without ever observing an absolute score.

**Algorithm, RLHF / PPO training loop**

```text
function RLHF(SFT_model, reward_model, prompts, beta, epsilon, n_iters):
    pi_theta     = copy(SFT_model)         # the trainable policy
    pi_ref       = freeze(SFT_model)       # KL anchor, never updated
    value_head   = init_value_head(SFT_model)

    for step in range(n_iters):
        x        = sample_batch(prompts)

        # ── 1. rollout: generate K completions per prompt
        y        = pi_theta.generate(x)                       # actions
        logp_new = pi_theta.log_prob(y | x)
        logp_old = stop_grad(logp_new)                        # snapshot
        logp_ref = pi_ref.log_prob(y | x)

        # ── 2. reward = preference score − KL penalty
        r_pref   = reward_model(x, y)                         # scalar per response
        kl       = logp_new - logp_ref                        # per-token
        r_total  = r_pref - beta * kl                         # shaped reward

        # ── 3. advantages via generalized advantage estimation
        V        = value_head(x, y)
        A        = compute_gae(r_total, V)

        # ── 4. PPO clipped objective + value loss
        ratio    = exp(logp_new - logp_old)
        L_policy = mean( min(ratio * A,
                              clip(ratio, 1-epsilon, 1+epsilon) * A) )
        L_value  = mse(V, returns(r_total))
        loss     = -L_policy + 0.5 * L_value

        optimizer.step(loss)
    return pi_theta
```

**Strengths**
- Optimizes directly for what we want (human preference), not a proxy
- Robust: the KL penalty and the clipped updates both guard against collapse
- Universally applicable: any scalar reward (preference, verifier, value head) plugs in

**Limitations & Failure Modes**
- Requires four models in memory: policy, reference, reward, value
- Reward hacking: policies that exploit reward-model blind spots
- Preference data is expensive; reward models inherit annotator biases
- High variance; many training runs diverge without careful tuning

### DPO: Direct Preference Optimization

**Rafailov et al., May 2023** | *RL-Free · Preference*
RLHF works, but the four-model dance is fragile and expensive. Rafailov and colleagues showed that the RLHF objective has a closed-form optimal policy in terms of the reward function:

$$
\pi^*(y \mid x) \;\propto\; \pi_{\text{ref}}(y \mid x) \cdot \exp\!\bigl(\,r(x, y)/\beta\,\bigr)
$$

*The optimal policy $\pi^*$ is the reference policy reweighted by exponentiated reward, with $\beta$ (the same KL coefficient as in RLHF) setting how sharply the reweighting acts. If we knew the reward function $r$, we could write the optimal policy down immediately; no rollouts would be needed.*

Inverting this relation gives an expression for the implicit reward in terms of $\pi^*$ and $\pi_{\text{ref}}$. Plugging that into the Bradley–Terry preference loss eliminates the reward model entirely, leaving a single supervised-style loss over preference pairs $(x, y_w, y_l)$:

$$
\mathcal{L}_{\text{DPO}}(\theta) = - \mathbb{E}\!\left[\,\log \sigma\!\left(\beta\, \log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta\, \log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\,\right]
$$

*No reward model, no PPO rollouts, no value head: just a contrastive loss on log-probability ratios. For each preference pair, the policy is pushed to raise the probability of the preferred response $y_w$ (relative to the reference model) and lower that of the rejected response $y_l$, with $\sigma$ again the logistic sigmoid and $\beta$ playing the same role as in RLHF.*

> **Insight:** DPO showed that much of the "magic" of RLHF could be replicated by a clever supervised reformulation. The open-source community embraced it (LLaMA-2-Chat-DPO, Tulu, Zephyr, early Qwen-Chat variants), and DPO became the default for community models in 2023–2024. Frontier labs still mostly use PPO or GRPO, because DPO is harder to extend to rewards that do not come as pairwise preferences, such as verifier outputs.

### GPT-4 and Tool-Augmented Reasoning

**OpenAI, March 2023** | *Multimodal · Tools*
GPT-4 was qualitatively a step up from GPT-3.5: graduate-level question answering, near-passing scores on the US bar exam, robust multi-step arithmetic, and (later) native image understanding. OpenAI did not publish the parameter count or architecture. What is on the record is that GPT-4 was trained with a substantially expanded SFT and RLHF stack, that it accepts text and image inputs natively, and that it was the model on which the harness, the tool-use product surface (function calling, code interpreter, browsing) was first widely deployed.

Two new ideas became prominent with GPT-4:

**Tool use**

A language model can call external functions (calculators, code interpreters, web search, vector databases) by emitting a structured request that the surrounding software executes. The tool's output is fed back into the model's context. This decouples reasoning from memorization: arithmetic is offloaded to a Python interpreter, and current events to a search engine.

**Multimodality**

Vision encoders convert images into token-like embeddings that sit in the same sequence as text tokens, so the same Transformer attends across both. This enables reasoning over diagrams, charts, and screenshots.

GPT-4's strength on reasoning came largely from scale and richer SFT/RLHF data, not from a new training paradigm. The next step (o1) would explicitly train the chain-of-thought itself.

### Tree of Thoughts & Search-Based Reasoning

**Yao et al., May 2023** | *Test-Time Search*
If chain-of-thought is a single linear trace, **Tree of Thoughts** generates many partial chains, scores them, and explores the promising branches the way a classical search algorithm would. The model plays both roles: *thought generator*, proposing candidate next steps, and *evaluator*, rating partial solutions. Backtracking becomes possible.

Variants quickly followed. Graph of Thoughts (Besta et al., 2024) generalized the tree to an arbitrary graph; Self-Refine (Madaan et al., 2023) cycles through drafting, critiquing, and revising; Reflexion (Shinn et al., 2023) maintains an episodic memory of past failures.

All of these are **test-time techniques over a fixed model**: they buy more reasoning for more compute, but the base model is not changed. They are the high-water mark of the "scaffolding" approach. From 2024 onward, the field would internalize much of this into model training itself.

## Era 4 — Reasoning by Reinforcement Learning with Verifiable Rewards (2024–2025)

*The decisive shift: instead of training on human preferences, train on problems whose answers can be checked automatically. This is reinforcement learning with verifiable rewards (RLVR). Reasoning is no longer prompted; it is learned.*

### OpenAI o1: Hidden Chain-of-Thought as a Trained Policy

**OpenAI, September 2024 (preview); December 2024 (full)** | *RLVR · Test-Time Scaling*
o1 was the first publicly available frontier model in which the chain-of-thought was no longer a prompting trick but a *trained behavior*. The model is fine-tuned with reinforcement learning to generate long, private "reasoning tokens" before its visible answer. On AIME, the American math olympiad qualifier described in the benchmarks section, o1 scored 83% versus GPT-4o's 13%; at the International Olympiad in Informatics 2024 it placed at the gold-medal threshold.

**The conceptual move**

- **Training data:** problems with checkable answers, drawn from math, competitive programming, and logic puzzles. The dataset is much smaller than a pretraining corpus, but the gradient signal is far more focused.

- **Objective:** reinforcement learning with the verifier as reward. The model is sampled at high temperature; rollouts that reach a correct final answer receive positive reward, regardless of the path.

- **Test-time scaling:** longer thought yields better answers. OpenAI explicitly demonstrated a *second scaling law*: accuracy improves roughly linearly with the logarithm of the number of reasoning tokens spent at inference, on top of the standard pretraining scaling law.

> **Insight:** o1 implicitly argued that the limits of "scaffolded" chain-of-thought were not fundamental: if the model is trained to use the scaffolding, it produces qualitatively better reasoning. The costs are opacity and price. o1's chain-of-thought is hidden from users, a controversial design choice, and inference is expensive, sometimes hundreds of thousands of reasoning tokens for a single problem.

o1's successor **o3** (April 2025) extended this to ARC-AGI, a benchmark explicitly designed to resist memorization. The ARC Prize evaluation reported two operating points: a high-efficiency configuration scoring 75.7 percent at roughly $26 per task, and a low-efficiency configuration scoring 87.5 percent at roughly $4,560 per task. The headline 87.5 percent was therefore not cheap, but the result completed the conceptual shift from "trained on text" to "trained on solving."

### DeepSeek-R1 and GRPO: Open Reasoning at Frontier Scale

**DeepSeek-AI, January 2025** | *GRPO · RLVR · Open Weights*
Four months after o1, DeepSeek released **R1**: a mixture-of-experts model with 671 billion total parameters (37 billion active per token), trained primarily by reinforcement learning on verifiable problems, with weights released under an MIT license. R1 matched o1 on most reasoning benchmarks, and its reasoning could be distilled into smaller dense models (Qwen-7B-R1, LLaMA-70B-R1) that retained much of the capability.

The technical highlight was **Group Relative Policy Optimization (GRPO)**, a variant of PPO that eliminates the value network entirely:

$$
L^{\text{GRPO}}(\theta) = \mathbb{E}\!\left[\,
        \min\!\Bigl(\rho_t(\theta)\, \hat{A}_t,\;
        \mathrm{clip}(\rho_t,\,1-\epsilon,\,1+\epsilon)\,\hat{A}_t\Bigr)\,\right] \;-\; \beta\, \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})
$$

*The clip-ratio structure is the same as PPO's, and the KL penalty against the reference policy $\pi_{\text{ref}}$ plays its usual role. The change is in the advantage estimate $\hat{A}_t$: it is computed from the *group* of responses to the same prompt, not from a learned value baseline.*

**Group-relative advantage**

For each prompt, the policy generates $G$ responses (typically 16). Each response receives a binary reward from a verifier: 1 if the math answer matches the ground truth, 0 otherwise. The advantage of response $i$ is its reward standardized against the group:

$$
\hat{A}_i = \frac{r_i - \mathrm{mean}(\{r_1, \dots, r_G\})}{\mathrm{std}(\{r_1, \dots, r_G\})}
$$

*Subtract the group's mean reward, divide by the group's standard deviation. Responses that beat the group average get a positive advantage and are upweighted; those below average are downweighted.*

No value network is trained; the group itself serves as the baseline. This is simpler, cheaper, and empirically more stable on problems where reward is sparse.

#### The R1-Zero discovery

DeepSeek first trained a model called **R1-Zero** by reinforcement learning alone, starting from a base pretrained model with *no supervised fine-tuning at all*. Against expectations, R1-Zero learned to produce coherent chains of thought and reached strong reasoning performance. Its chains were sometimes hard to read (mixed languages, no formatting), so the final R1 was bootstrapped with a small supervised round on curated traces before resuming reinforcement learning. The headline finding stood: **reasoning emerged from reinforcement learning on verifiable rewards alone**, and it has reshaped the field.

**Algorithm, GRPO step**

```text
function GRPO_step(policy, ref_policy, verifier, prompts, G, epsilon, beta):
    x = sample(prompts)

    # 1. Generate G responses to the same prompt
    Y           = [policy.generate(x) for _ in range(G)]
    logp_new    = [policy.log_prob(y | x) for y in Y]
    logp_old    = stop_grad(logp_new)
    logp_ref    = [ref_policy.log_prob(y | x) for y in Y]

    # 2. Verifier produces binary reward for each response
    r           = [verifier.check(x, y) for y in Y]            # 0 or 1

    # 3. Group-relative advantage, no value network
    A           = [(r[i] - mean(r)) / (std(r) + 1e-6) for i in range(G)]

    # 4. PPO-clip objective + KL penalty
    ratio       = [exp(logp_new[i] - logp_old[i]) for i in range(G)]
    L_policy    = mean([min(ratio[i] * A[i],
                             clip(ratio[i], 1-epsilon, 1+epsilon) * A[i])
                        for i in range(G)])
    L_kl        = mean([logp_new[i] - logp_ref[i] for i in range(G)])
    loss        = -(L_policy - beta * L_kl)
    return loss
```

> **Insight:** GRPO is now the dominant algorithm for reasoning-focused reinforcement learning in the open-source ecosystem. Qwen3, the Llama-3.x reasoning variants, and most academic reasoning models use either GRPO or one of its close cousins (RLOO, REINFORCE++).

### Qwen3: Open Hybrid Reasoning

**Alibaba, May 2025** | *Hybrid Reasoning · Open Weights*
Qwen has evolved through Qwen-1 (mid-2023), Qwen-1.5 (Feb 2024), Qwen2 (June 2024), Qwen2.5 (Sep 2024), and Qwen3 (May 2025). Each release brought architectural and training upgrades; Qwen3 is the first that bundles reasoning RL into the open release.

**What Qwen3 changed**

- **Hybrid thinking mode.** A single model supports two inference modes selectable by a simple control token: a *fast* non-thinking mode (chat-style, short responses) and a *thinking* mode that emits a private chain-of-thought before the answer, in the o1/R1 style. Users can toggle per query.

- **Mixture-of-experts at scale.** The flagship Qwen3-235B-A22B activates 22 billion parameters per token out of 235 billion total, giving cheaper inference than dense models of comparable quality.

- **A verifiable-rewards pipeline.** A four-stage training recipe: pretraining, then supervised fine-tuning on long chain-of-thought examples as a cold start, then reasoning-focused reinforcement learning with verifiable rewards on math and code, then a final round of general preference tuning in the RLHF style.

- **Native 128K-token context**, and multilingual coverage of 119 languages and dialects, with Chinese, English, and code as the primary reasoning languages.

The Qwen line illustrates the conceptual shifts compactly. Qwen-1 was a GPT-3-style stack of pretraining, supervised fine-tuning, and simple RLHF. Qwen2.5 added stronger math and code data and a polished combination of DPO and RLHF. Qwen3 introduced verifiable-reward reinforcement learning and the dual-mode reasoning interface. One family, with a single provenance and tokenizer lineage, thus embodies all three of the big training-paradigm shifts of the last three years.

### GPT-5: Routing Between Fast and Slow

**OpenAI, August 2025** | *Hybrid Reasoning · Routing*
GPT-5 unified the GPT and o lines. Rather than two product surfaces (GPT-4o for chat, o-series for reasoning), GPT-5 routes each query at the system level between a fast non-reasoning core and a deeper reasoning policy. The router itself is a learned classifier; users can override with a "thinking" toggle.

From the standpoint of this review, the interesting thing about GPT-5 is not its architecture, whose details remain confidential, but the way it turned the two-mode idea into a product, in parallel with Qwen3. For most prompts you want a one-pass response; a small fraction of queries deserve hundreds of thousands of reasoning tokens. Routing makes that distinction at inference time without burdening the user.

> **Insight:** From a research standpoint, GPT-5 and Qwen3 mark the maturation of the verifiable-rewards paradigm into deployed product. Both companies separately concluded that reinforcement learning on long chains of thought is necessary for top reasoning performance, that test-time compute scaling complements pretraining scaling, and that users want to choose between speed and depth.

## Era 5 — Latent & Recursive Reasoning (2023–2026)

*A parallel track. Reasoning depth is decoupled from output length; computation happens inside a recurrent latent state. Culminates in the paper that motivated this review: GRAM.*

### Looped Transformers

**Yang, Lee, Nowak, Papailiopoulos, November 2023** | *Weight Sharing · Algorithmic*
Yang and colleagues revisited the Universal Transformer in a new light: what if a small set of Transformer blocks is applied repeatedly, with the explicit goal of *learning algorithms* (sorting, copying, regression, arithmetic) rather than modeling language? They showed that looped Transformers can simulate algorithmic procedures with very few parameters, and that the loop count can be varied at inference to trade compute for accuracy.

$$
h^{(t+1)} = \text{Block}_\theta\!\bigl(h^{(t)}\bigr), \quad t = 0, 1, \dots, T-1
$$

*$h^{(t)}$ is the hidden state after $t$ applications of the block, and $\theta$ is the block's one fixed set of parameters, applied $T$ times in total. Increasing $T$ increases effective depth without adding parameters.*

Looped Transformers reframed recurrence as a deliberate architectural choice for reasoning tasks. They underperformed standard language models on language itself, but excelled on structured-reasoning probes, exactly the niche the recursive-reasoning lineage now occupies.

### Coconut: Reasoning in Continuous Latent Space

**Hao, Sukhbaatar, Su, Li, Hu, Weston, Tian, December 2024** | *Continuous Thought · Latent CoT*
Chain-of-thought commits the model to verbalizing each intermediate step as discrete tokens. Coconut ("Chain Of CONtinUous Thought") proposed a hybrid: instead of decoding the next token at each reasoning step, feed the model's *last hidden state* directly back in as the next input embedding. The model "thinks" in the same continuous space its hidden states inhabit, then switches back to token generation to produce the final answer.

Training Coconut required a curriculum that gradually replaces the first few chain-of-thought tokens with their hidden-state equivalents. Hao and colleagues found that Coconut matched or exceeded written-out chains of thought on logical-reasoning tasks while emitting far fewer visible tokens, suggesting that internal latent reasoning can be more compact than reasoning written out as text. This is one of several recent threads (*Reasoning by Superposition*, Zhu et al. 2025; *Continuous Chain of Thought Enables Parallel Exploration*, Gozeten et al. 2025) arguing for latent rather than token-level reasoning.

### HRM: Hierarchical Reasoning Model

**Wang, Li, Sun, Chen, Liu, Wu, Lu, Song, Abbasi Yadkori, June 2025** | *Hierarchical · Deep Supervision*
HRM is the first recursive reasoning model to seriously challenge large language models on hard structured tasks like Sudoku-Extreme and ARC-AGI, with a very compact 27-million-parameter architecture and no pretraining on internet text. The key innovations:

**Architectural elements**

- **A two-level latent state $z = (h, l)$:** a *high-level* component $h$ that carries abstract reasoning state across many steps, and a *low-level* component $l$ that performs fine-grained intermediate computation. The two operate at different time scales, an echo of the fast and slow systems in dual-process theories of human cognition.

- **Nested recursion:** at each high-level transition, the low-level component is iterated $K$ times before $h$ is updated. With $T$ high-level transitions, each supervision step runs $T \times K$ low-level updates in total.

- **Deep supervision:** the decoder is invoked after every supervision step and the training loss is applied each time, propagating dense gradients through long chains of recursion.

- **Adaptive computation time:** a learned halting head decides, per problem instance, how many supervision steps to run, in the spirit of the Universal Transformer.

HRM reaches 55% on Sudoku-Extreme and 40.3% on ARC-AGI-1 from scratch: no pretraining, no instruction tuning, no reinforcement learning. It is a remarkably clean demonstration that architectural recurrence alone, with the right hierarchical structure and deep supervision, can match or exceed much larger models on constrained reasoning benchmarks.

> **Insight:** HRM is deterministic. The same prompt always produces the same answer. This is fine when there is a unique solution; it fails badly on multi-solution problems (N-Queens, graph coloring) where the model collapses to one of the valid solutions and cannot explore the rest. That limitation will motivate GRAM.

### TRM: Less is More with Tiny Recursive Networks

**Jolicoeur-Martineau, October 2025** | *Compact · Recursive*
If HRM showed that 27 million parameters could do something a 100-billion-parameter model could not, TRM pushed the lesson to its limit. With a 7-million-parameter network, several orders of magnitude smaller than frontier language models, TRM matches or exceeds HRM on Sudoku-Extreme (87.4%) and ARC-AGI-1 (44.6%) by leaning even more heavily on recursive depth.

**TRM's simplifications and gains**

- **Smaller is faster.** Fewer parameters make each recursion step cheaper, which buys more steps within the same training budget. TRM trains with longer recursion than HRM at lower wall-clock cost.

- **Simplified hierarchy.** TRM keeps the two-component latent state but uses a more uniform update rule.

- **Truncated gradient propagation.** Gradients are backpropagated only through the final transition of each supervision step. This is memory-efficient and, surprisingly, does not hurt accuracy.

- **Same deterministic limitation.** Like HRM, TRM follows a single fixed latent trajectory and converges to a single prediction. Multi-solution coverage remains poor.

TRM's title, "Less is More," captures the bet: with the right inductive bias (recursion plus a hierarchical state) and the right training (deep supervision), compact networks can solve problems that frontier language models struggle with. It is the strongest recent evidence that scale is not the only route to reasoning.

### GRAM: Generative Recursive Reasoning Models

**Baek, Jo, Kim, Ren, Bengio, Ahn, May 2026** | *Generative · Variational · Multi-Trajectory*
HRM and TRM established recursion as a viable reasoning paradigm, but their determinism is a structural ceiling. **GRAM** (Generative Recursive reAsoning Models) keeps the recursive backbone and makes each latent transition *stochastic* by adding a learned Gaussian perturbation on top of the deterministic update. The result is a latent-variable generative model over reasoning trajectories: the model defines a probability distribution over paths of internal states, rather than a single fixed path.

#### The stochastic transition

At each recursion step, GRAM first computes the deterministic high-level update $u_t$ as HRM/TRM would. It then samples a state-dependent Gaussian perturbation and adds it:

$$
\epsilon_t \sim p_\theta(\epsilon_t \mid u_t) := \mathcal{N}\!\bigl(\mu_\theta(u_t),\, \sigma^2_\theta(u_t)\, I\bigr), \qquad
      h_t = u_t + \epsilon_t
$$

*The perturbation $\epsilon_t$ is drawn from a Gaussian whose mean $\mu_\theta(u_t)$ and variance $\sigma^2_\theta(u_t)$ are computed from the current update $u_t$ by small learned networks ($I$ is the identity matrix, so each latent dimension gets independent noise). The mean acts as a learned guidance direction in latent space; the variance controls how widely the model explores. The low-level component $l$ remains deterministic; randomness is injected only at the abstract high level.*

#### The training objective: amortized variational inference

Because the trajectory of latent states $\tau = (z_0 \to z_1 \to \dots \to z_T)$ is never observed, GRAM cannot directly maximize $\log p_\theta(y \mid x)$, the log-probability of the correct answer $y$ given the input $x$: computing it would require summing over every possible trajectory. It instead maximizes an evidence lower bound (ELBO), the standard workaround in latent-variable modeling:

$$
\log p_\theta(y \mid x) \;\ge\;
        \mathbb{E}_{q_\phi(\tau \mid x, y)}\!\bigl[\log p_\theta(y \mid \tau, x)\bigr]
        \;-\; \mathrm{KL}\!\bigl(q_\phi(\tau \mid x, y)\,\big\|\,p_\theta(\tau \mid x)\bigr)
$$

*Two ingredients. The variational posterior $q_\phi(\tau \mid x, y)$, with its own parameters $\phi$, gets to peek at the target answer $y$ and proposes trajectories likely to reach it; the first term rewards trajectories that decode to the right answer. The prior $p_\theta(\tau \mid x)$ sees only the input and is what runs at inference time. The KL term penalizes the gap between the two, so that trajectories sampled from the prior at inference resemble the good ones found during training.*

Both prior and posterior are Markov processes over latent states: each new state depends only on the current one. They share the deterministic transition module and differ only in their noise distributions. The posterior uses its knowledge of $y$ to find trajectories that lead to correct answers; the prior must learn to find such trajectories without that knowledge.

#### Width-based inference-time scaling

Because each forward pass is stochastic, running GRAM multiple times on the same input produces *different* reasoning trajectories. This opens a new test-time scaling axis: not just deeper recursion (a larger number of steps $T$) but *wider* sampling (a larger number of independent trajectories $N$).

The candidate outputs from the $N$ trajectories are aggregated in one of two ways. **Majority voting** picks the most common answer, analogous to self-consistency in chain-of-thought prompting. Alternatively, a **Latent Process Reward Model (LPRM)**, a small value head trained to predict final accuracy from the terminal latent state, scores each trajectory, and the best-scoring one wins. The latter gives GRAM the equivalent of a process reward model, but learned directly on latent states rather than on text.

> **Insight:** The Latent Process Reward Model is GRAM's clearest connection to the reinforcement learning lineage. Just as o1 and R1 use verifier rewards to weight chain-of-thought rollouts, GRAM uses a learned value head to select among latent rollouts. The reward signal is consumed differently, in selection rather than in a policy-gradient update, but the conceptual move is the same: learn a quality function once, then apply it across many samples.

#### Empirical highlights

- On **Sudoku-Extreme**: 97.0% for GRAM, versus 87.4% for TRM, 55.0% for HRM, and 61.3% for the Looped Transformer.

- On **ARC-AGI-1**: 55.7% versus 52.0% (TRM) and 44.6% (HRM); competitive with much larger reasoning language models.

- On **8×8 N-Queens**: 99.7% accuracy with 90.3% coverage of all valid solutions, versus TRM's 66.8% accuracy and 36.1% coverage. Deterministic recursion fundamentally cannot capture multiple solutions.

- On **unconditional generation of MNIST digit images**: a Fréchet Inception Distance of 73.34 with 256 steps, comparable to D3PM (a discrete diffusion model) despite GRAM's network having only 10 million parameters.

- GRAM drawing 20 samples at 16 recursion steps outperforms TRM at 320 recursion steps (97.0% versus 90.5%) at comparable compute. Width can beat depth.

**Algorithm, GRAM training step (truncated ELBO surrogate)**

```text
function GRAM_step(prior_pθ, posterior_qϕ, decoder, encoder, x, y, T, K, N_sup):
    # 1. Encode the input once; reused throughout the recursion
    e_x      = encoder(x)
    h        = zeros(d_h)
    l        = zeros(d_l)

    for sup_step in range(N_sup):
        # 2. Inner loop: T high-level transitions; K low-level refinements per transition
        for t in range(T):
            # Low-level deterministic refinement
            for k in range(K):
                l = f_L(h, l, e_x)

            # High-level deterministic update u_t
            u = f_H(h, l)

            # Sample from posterior (uses target y); prior is matched in KL
            mu_q,   sig_q   = posterior_qϕ(u, y)
            mu_p,   sig_p   = prior_pθ(u)
            eps             = reparam_gaussian(mu_q, sig_q)
            h               = u + eps

        # 3. Decode terminal high-level state for this supervision step
        y_hat   = decoder(h)
        ce_loss = cross_entropy(y_hat, y)

        # 4. KL between posterior and prior at the final transition only (truncated)
        kl      = kl_gaussian(mu_q, sig_q, mu_p, sig_p)

        loss_sup = ce_loss + kl
        backprop(loss_sup)
        # gradient is propagated only through the last transition (memory-efficient)
        h, l = stop_grad(h), stop_grad(l)

    return loss
```

**Strengths**
- Recovers multi-solution structure (90% coverage on N-Queens versus under 36% for deterministic recursion)
- Width-based inference scaling is parallelizable, with no autoregressive bottleneck
- One framework covers both conditional reasoning $p_\theta(y\mid x)$ and unconditional generation $p_\theta(x)$
- Stochastic guidance is a drop-in extension, with consistent gains on every architecture tested
- Compact: 10 million parameters, competitive with frontier reasoning models on structured benchmarks

**Limitations & Failure Modes**
- The truncated-gradient surrogate is biased; the full ELBO is intractable for long trajectories
- Deep supervision is inherently sequential, so training is harder to parallelize than standard Transformer pretraining
- Scaling to the foundation-model regime (billions of parameters, internet text) is unproven
- Reward-model-based selection works on tasks with automatic graders; it is less clear how it applies to open-ended generation

---

## Comparison Matrix

Reasoning models differ along many axes: the data they train on, what they optimize, how they spend compute at inference, and whether they can explore multiple solutions. The table below captures the conceptual landscape.

| Model / Method | Year | Params | Pretraining Data | Fine-Tune Signal | Inference Mode | Multi- Solution | Open Weights |
|---|---|---|---|---|---|---|---|
| _Era 1, Foundations_ |  |  |  |  |  |  |  |
| Transformer | 2017 | ~65M | WMT (parallel) | Sup. translation | One pass | ✗ | ✓ |
| GPT-1 / BERT | 2018 | 117M / 340M | Books / Wiki | CE on text | One pass | ✗ | ✓ |
| Universal Transformer | 2018 | ~10M | Algorithmic | CE + ACT | Recursive | ✗ | ✓ |
| _Era 2, Scale & First Reasoning_ |  |  |  |  |  |  |  |
| GPT-3 | 2020 | 175B dense | Web | CE only | One pass + CoT prompting | ~ | ✗ |
| Qwen-1 | 2023 | 7B–72B | Multilingual web | SFT + RLHF | One pass + CoT | ~ | ✓ |
| ChatGPT (GPT-3.5) | 2022 | ~175B | Web | SFT + RLHF | One pass | ~ | ✗ |
| _Era 3, Alignment by RL_ |  |  |  |  |  |  |  |
| GPT-4 | 2023 | undisclosed | Web + multimodal | SFT + RLHF + tools | One pass + CoT + tools | ~ | ✗ |
| DPO models | 2023 | 7B–70B | Web | DPO (no RM) | One pass | ~ | ✓ |
| _Era 4, RL with Verifiable Rewards_ |  |  |  |  |  |  |  |
| OpenAI o1 | 2024 | ? | Web + reasoning | RLVR (PPO) | Hidden long-CoT | ~ | ✗ |
| DeepSeek-R1 | 2025 | 671B (37B active) | Web + math/code | RLVR (GRPO) | Visible long-CoT | ~ | ✓ |
| Qwen3 | 2025 | 0.6B–235B MoE | Multilingual web + reasoning | RLVR + RLHF | Hybrid (toggle) | ~ | ✓ |
| GPT-5 | 2025 | ? | Web + reasoning | RLVR + RLHF + router | Routed fast / slow | ~ | ✗ |
| _Era 5, Latent & Recursive Reasoning_ |  |  |  |  |  |  |  |
| Looped Transformer | 2023 | ~1M | Algorithmic (sup.) | CE only | Recursive | ✗ | ✓ |
| Coconut | 2024 | 0.5B–7B | Reasoning curric. | CE on hybrid CoT | Latent loop + token | ~ | ✓ |
| HRM | 2025 | 27M | Task-specific | Deep supervision (CE) | Hierarchical recursive | ✗ | ✓ |
| TRM | 2025 | 7M | Task-specific | Truncated deep sup. | Recursive | ✗ | ✓ |
| **GRAM** | **2026** | 10M | Task-specific | ELBO + LPRM | Stochastic recursive + parallel | ✓ | ✓ |

✓ = yes ✗ = no ~ = partial / via sampling | *CE* = cross-entropy · *CoT* = chain-of-thought · *RM* = reward model · *RLVR* = reinforcement learning with verifiable rewards · *MoE* = mixture of experts

---

## Open Challenges

**Reward hacking**

Reinforcement learning on verifiable rewards is only as good as the verifier. Policies have been observed to discover exploits: manipulating unit tests, writing answers that fool checkers, or "thinking" in adversarial ways that satisfy the reward but not the intent behind it.

**Faithfulness of hidden chains of thought**

o1, o3, and GPT-5 emit private reasoning. Whether the visible answer actually *uses* that reasoning, or whether the reasoning is a post-hoc rationalization, is unclear. Mechanistic interpretability studies are still catching up.

**Scaling recursion**

HRM, TRM, and GRAM excel at structured tasks but have not yet been scaled to foundation-model regimes. Sequential deep supervision is hard to parallelize; the bottleneck is engineering, not just compute.

**Compute economics**

A single o3 solution can cost tens to thousands of dollars depending on the configuration. Width-scaling models like GRAM spend their compute differently but still pay a test-time tax. Cheaper reasoning per problem is an open practical frontier.

**Reasoning beyond verifiable domains**

Math and code have checkers. Law, medicine, ethics, and science writing do not. Whether reinforcement learning with verifiable rewards generalizes to these soft-reward domains, or whether new reward sources are needed, is the next paradigm question.

**Bridging the two tracks**

Token-based reasoning (o1, R1) and latent recursive reasoning (HRM, TRM, GRAM) have not yet been combined at scale. Would a Transformer with a recursive reasoning block, trained on verifiable rewards, deliver the best of both? The question is open.

---

## Summary: The Evolution of Reasoning Research

The decade compresses into four moves:

1. Pretrain a Transformer on web text. Reasoning appears as an accidental side effect of scale.

2. Fine-tune on human preferences with RLHF. Chain-of-thought emerges as a prompting trick.

3. Train the chain-of-thought itself with reinforcement learning on verifiable rewards. Test-time scaling becomes a law.

4. Move the reasoning into a recurrent latent state. Reasoning depth decouples from output length, and trajectories become probabilistic.

> **Insight:** The story of reasoning models is the story of who provides the gradient. Pretraining lets next-token statistics carry the signal. RLHF hands the gradient to human preference. Verifiable-reward training hands it to a checker. Recursive models like GRAM hand it to a posterior over latent trajectories. The architectures change; the deeper movement is the migration of the supervisory signal, from passive text, to active humans, to formal checkers, to the model's own latent self-evaluation.

GRAM, the paper that motivated this review, sits at the intersection of two arcs. From the recursive-reasoning lineage it inherits compact, deep computation through shared transition functions. From the generative-modeling lineage it inherits stochastic latent trajectories trained by amortized variational inference. The combination is a generative model of reasoning itself, and a credible answer to the question its authors pose in their first paragraph: "How should future neural reasoning systems implement extended computation?" The answer GRAM offers is not *longer chains* but *more of them in parallel, each refined recursively*.

---

## References

1. **Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin** (2017). Attention Is All You Need. *NeurIPS*.

2. **Radford, Narasimhan, Salimans, Sutskever** (2018). Improving Language Understanding by Generative Pre-Training. *OpenAI*.

3. **Devlin, Chang, Lee, Toutanova** (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL*.

4. **Raffel, Shazeer, Roberts, Lee, Narang, Matena, Zhou, Li, Liu** (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5). *JMLR*.

5. **Dehghani, Gouws, Vinyals, Uszkoreit, Kaiser** (2019). Universal Transformers. *ICLR*.

6. **Graves** (2016). Adaptive Computation Time for Recurrent Neural Networks. *arXiv:1603.08983*.

7. **Brown, Mann, Ryder et al.** (2020). Language Models are Few-Shot Learners (GPT-3). *NeurIPS*.

8. **Bai, Bai, Chu et al.** (2023). Qwen Technical Report. *arXiv:2309.16609*.

9. **Shazeer** (2020). GLU Variants Improve Transformer. *arXiv:2002.05202*.

10. **Su, Lu, Pan, Murtadha, Wen, Liu** (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. *arXiv:2104.09864*.

11. **Zhang, Sennrich** (2019). Root Mean Square Layer Normalization. *NeurIPS*.

12. **Ba, Kiros, Hinton** (2016). Layer Normalization. *arXiv:1607.06450*.

13. **Hendrycks, Gimpel** (2016). Gaussian Error Linear Units (GELU). *arXiv:1606.08415*.

14. **Ramachandran, Zoph, Le** (2017). Searching for Activation Functions (Swish). *arXiv:1710.05941*.

15. **Peng, Quesnelle, Fan, Shippole** (2023). YaRN: Efficient Context Window Extension of Large Language Models. *arXiv:2309.00071*.

16. **Yang, Yang, Zhang et al.** (2024). Qwen2.5 Technical Report. *arXiv:2412.15115*.

17. **Qwen Team** (2025). Qwen3 Technical Report. *Alibaba Cloud*.

18. **Wei, Wang, Schuurmans, Bosma, Xia, Chi, Le, Zhou** (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS*.

19. **Wang, Wei, Schuurmans, Le, Chi, Narang, Chowdhery, Zhou** (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *ICLR*.

20. **Yao, Yu, Zhao, Shafran, Griffiths, Cao, Narasimhan** (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. *NeurIPS*.

21. **Besta, Blach, Kubicek et al.** (2024). Graph of Thoughts: Solving Elaborate Problems with Large Language Models. *AAAI*.

22. **Christiano, Leike, Brown, Martic, Legg, Amodei** (2017). Deep Reinforcement Learning from Human Preferences. *NeurIPS*.

23. **Schulman, Wolski, Dhariwal, Radford, Klimov** (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.

24. **Williams** (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. *Machine Learning*.

25. **Ouyang, Wu, Jiang et al.** (2022). Training Language Models to Follow Instructions with Human Feedback (InstructGPT). *NeurIPS*.

26. **OpenAI** (2022). ChatGPT: Optimizing Language Models for Dialogue. *openai.com*.

27. **OpenAI** (2023). GPT-4 Technical Report. *arXiv:2303.08774*.

28. **Rafailov, Sharma, Mitchell, Ermon, Manning, Finn** (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *NeurIPS*.

29. **OpenAI** (2024). Learning to Reason with LLMs (o1). *openai.com*.

30. **OpenAI** (2025). Introducing GPT-5. *openai.com*.

31. **DeepSeek-AI** (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. *arXiv:2501.12948*.

32. **Shao, Wang, Zhu et al.** (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. *arXiv:2402.03300*.

33. **Yang, Lee, Nowak, Papailiopoulos** (2023). Looped Transformers are Better at Learning Learning Algorithms. *arXiv:2311.12424*.

34. **Hao, Sukhbaatar, Su, Li, Hu, Weston, Tian** (2024). Training Large Language Models to Reason in a Continuous Latent Space (Coconut). *arXiv:2412.06769*.

35. **Zhu, Hao, Hu, Jiao, Russell, Tian** (2025). Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought. *arXiv:2505.12514*.

36. **Gozeten, Ildiz, Zhang, Harutyunyan, Rawat, Oymak** (2025). Continuous Chain of Thought Enables Parallel Exploration and Reasoning. *arXiv:2505.23648*.

37. **Wang, Li, Sun, Chen, Liu, Wu, Lu, Song, Abbasi Yadkori** (2025). Hierarchical Reasoning Model (HRM). *arXiv:2506.21734*.

38. **Jolicoeur-Martineau** (2025). Less is More: Recursive Reasoning with Tiny Networks (TRM). *arXiv:2510.04871*.

39. **Baek, Jo, Kim, Ren, Bengio, Ahn** (2026). Generative Recursive Reasoning Models (GRAM). *arXiv:2605.19376*.

40. **Chollet** (2019). On the Measure of Intelligence (ARC). *arXiv:1911.01547*.

41. **Chollet, Knoop, Kamradt, Landers, Pinkard** (2025). ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems. *arXiv:2505.11831*.

42. **Cobbe, Kosaraju, Bavarian et al.** (2021). Training Verifiers to Solve Math Word Problems (GSM8K). *arXiv:2110.14168*.

43. **Hendrycks, Burns, Kadavath et al.** (2021). Measuring Mathematical Problem Solving with the MATH Dataset. *NeurIPS Datasets and Benchmarks*.

44. **Hendrycks, Burns, Basart et al.** (2020). Measuring Massive Multitask Language Understanding (MMLU). *arXiv:2009.03300*.

45. **Chen, Tworek, Jun et al.** (2021). Evaluating Large Language Models Trained on Code (Codex / HumanEval). *arXiv:2107.03374*.

46. **Srivastava, Rastogi, Rao et al.** (2022). Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models (BIG-Bench). *arXiv:2206.04615*.

47. **Suzgun, Scales, Schärli et al.** (2022). Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them (BBH). *arXiv:2210.09261*.

48. **LeCun, Bottou, Bengio, Haffner** (1998). Gradient-Based Learning Applied to Document Recognition (MNIST). *Proceedings of the IEEE*.

49. **Salimans, Goodfellow, Zaremba, Cheung, Radford, Chen** (2016). Improved Techniques for Training GANs (Inception Score). *NeurIPS*.

50. **Heusel, Ramsauer, Unterthiner, Nessler, Hochreiter** (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID). *NeurIPS*.

51. **Kahneman** (2011). Thinking, Fast and Slow. *Farrar, Straus and Giroux*.

52. **Bengio** (2017). The Consciousness Prior. *arXiv:1709.08568*.

53. **Sutton, Barto** (2018). Reinforcement Learning: An Introduction (2nd ed.). *MIT Press*.