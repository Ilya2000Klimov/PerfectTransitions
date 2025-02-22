---
layout: default
title: Status
---
## Project Summary

<!-- A short paragraph outlining the main idea of the project (see further
instructions here: https://royf.org/crs/CS175/W25/proposal.pdf). You now have
a better sense of what your project is about than you did in the proposal, so update and clarify
beyond that version. Do not change your proposal page! (A few of you have already changed
it to reflect a change of topic, but don’t change it further, rather keep it as it was at the time
you locked down your topic). -->

This project focuses on creating a **seamless music transition model** that predicts the best next segment of a song based on learned embeddings. The goal is to generate **natural-sounding transitions** between music clips without abrupt changes.

The approach consists of three main components:

1. **Feature Extraction:** Using **BEATs Iteration 3** (a Transformer-based model for audio understanding) to extract **time-step embeddings** from songs.
2. **LSTM Training:** A **Bidirectional LSTM** trained with **contrastive triplet loss**, ensuring that the model learns to **smoothly transition between music segments**.
3. **Transition Matching:** Computing **cosine similarity** between embeddings to **predict the best transition point** between songs.

The dataset consists of **full-length songs** (from the **FMA dataset**), which are preprocessed by **segmenting them into 20-30 second clips** with **overlapping context** to preserve musical continuity. The final goal is to deploy this model for **automatic DJing, playlist blending, and AI-generated song transitions**.


## Approach
### **1. Feature Extraction Using BEATs Iteration 3**

We use **BEATs-Large (Iteration 3, AS2M model)** to extract **per-frame embeddings** from segmented audio clips.

- Songs are **resampled to 16 kHz** and converted to **mono**.
- BEATs extracts embeddings at each **time-step (T × D format)**.
- **L2-normalization** is applied before passing embeddings to the LSTM.
- Data is stored in `.npy` format for fast retrieval.

**Implementation Steps:**

- Songs are cut into **overlapping** 20-30 second segments (default: **25s clips, 5s overlap**).
- Segments are **named sequentially** to maintain their ordering.
- Embeddings are extracted and stored in **/NFS/dataset/embeddings/**.
- A **checkpointing system** ensures the process resumes from where it left off if interrupted.

### **2. Training the LSTM for Transition Prediction**

We train a **Bidirectional LSTM** with **contrastive triplet loss** to predict the most seamless transition between song clips.

- **Two-Stage Embeddings**: The **end of one segment (A)** is matched with the **start of another segment (B)**.
- **Triplet Loss Objective:**
  $$
  \mathcal{L}_{triplet} = max(0, d(A,B) - d(A,C) + margin)
  $$
  where **(A, B, C)** is a triplet of (anchor, positive, negative) clips.
- **Exponential Weighting**:
  - The **last 3-5s** of segment A and the **first 3-5s** of segment B are given **higher weight** to prioritize smooth transitions.
- **Hyperparameters:**
  - **Hidden Units:** 128
  - **Layers:** 2-3
  - **Dropout:** 0.2
  - **Loss Margin:** 0.2-0.4
  - **Batch Size:** 32
  - **Optimizer:** Adam
  - **Learning Rate Schedule:** Cosine Annealing

Training is performed on **SLURM** using GPU nodes, with **checkpointing enabled** for preemption handling.

### **3. Transition Matching & Deployment**

After training, **cosine similarity** is used to **rank candidate transitions**:

- The model takes a clip’s **final embedding** and finds the **best next segment**.
- **FAISS (Facebook AI Similarity Search)** is used for **efficient retrieval**.
- **ONNX & PyTorch JIT** are explored for **fast inference deployment**.

### **4. Evaluation Metrics**

We evaluate **both perceptual and mathematical** similarity metrics:

- **Spectral Convergence & Mel-Spectrogram MSE**: Measures **audio continuity**.
- **Cosine Similarity & Triplet Loss**: Evaluates **embedding consistency**.
- **User Listening Tests**: Collect subjective ratings of transition quality.

<img src="./res/Project_outline.jpg" alt="Project Outline" width="852" height="602">

<!-- Give a detailed description of your approach, in a few of paragraphs (at least a
couple). You should summarize the main algorithm you are using, such as by writing out how
it samples data and the equations of the loss(es) it optimizes (you can copy this information
from scientific publications or online resources, in which case cite them clearly. The default
GitHub Pages we shared includes an example of redering math within Markdown). You
should also give details about the approach as it applies to your scenario, such as how you set
up inputs and outputs (e.g. states / observations, actions, and rewards), how much data you
use (e.g. for how many interaction steps you train), and the values of any hyperparameters
(cite your source for default hyperparameter values, and for any changed values detail if and
how you tune them and the numbers you end up using). A good guideline is to incorporate
sufficient details so that most of your approach is reproducible by a reader. You're encouraged
to use figures for this, as appropriate, e.g. as we used in the exercises. -->

## Evaluation

<!-- An important aspect of your project, as we mentioned in the beginning, is
evaluating your project. Be clear and precise about describing the evaluation setup, for both
quantitative and qualitative results. Present the results to convince the reader that you have a
working implementation. Use plots, charts, tables, screenshots, figures, etc. as appropriate.
For each type of evaluation that you perform, you'll probably need at least 1 or 2 paragraphs
(excluding figures etc.) to describe it. -->

We assess both **quantitative model accuracy** and **qualitative smoothness of music transitions**.

### **Quantitative Evaluation**

- **Triplet Loss Curve:** Monitor loss decrease over training epochs.
- **Cosine Similarity Distribution:** Compare the similarity of correct vs. incorrect transitions.
- **LSTM Loss Convergence:** Validate that the model improves over time.

### **Qualitative Evaluation**

- **Random vs. Learned Transitions:** Compare our model’s transitions to **randomly shuffled segments**.
- **Spectrogram Alignment:** Ensure frequency-domain continuity.
- **Listening Study:** Have users rate **seamless vs. jarring transitions**.

All evaluations are **logged in Weights & Biases (W&B)** for experiment tracking.


## Remaining Goals and Challenges
<!-- In a few paragraphs, describe your goals for the remainder
of the quarter. At the very least, describe how you consider your prototype to be limited,
and what you want to add to make it a complete contribution. Note that if you think your
algorithm is working well enough, but have not performed sufficient evaluation to gain insight,
doing this should be a goal. Similarly, you may propose comparing with other algorithms
/ approaches / manually tailored solutions (when feasible) that you did not get a chance to
implement, but can enrich your discussion in the final report. Finally, given your experience
so far, describe some of the challenges you anticipate facing by the time your final report is
due, to what degree you expected to become obstacles, and what you might try in order to
overcome them. -->

### **Two-Stage Embedding (Separate Start & End)**

- Optimize **embedding separation** for better **sequence learning**.

### **Predicting Next Music Segment for Seamless Transitions**

- Introduce a **diffusion model** for transition refinement.

### **Optimizing Crossfade Durations to Find the Best Seamless Song Transition**

- Fine-tune the **best fade-in/fade-out durations** for **seamless audio blending**.

### **Interactive Interface**

- Create a **demo UI** to **visualize transitions** and allow user feedback.

### **Music App Integration**

- Test deployment into **Spotify/YouTube DJ playlists** or **local audio mixing software**.


## Resources Used

<!-- Mention all the resources that you found useful in writing your implemen-
tation. This should include everything like code documentation, AI/ML libraries, source
code that you used, StackOverflow, etc. You do not have to be comprehensive, but it is
important to report the sources that are crucial to your project. One aspect that does need to
be comprehensive is a description of any use you made of AI tools. -->

### **Research Papers & Code**

- **BEATs Transformer Model** (Microsoft UniLM):
  - [GitHub](https://github.com/microsoft/unilm/tree/master/beats)
  - **Pre-trained Checkpoints**: BEATs Iteration 3 AS2M
- **Triplet Loss**:
  - Hoffer & Ailon (2015): “Deep Metric Learning Using Triplet Network”
  - [Faiss: Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
  - [Shazam Music Processing Fingerprinting and Recognition] (https://www.toptal.com/algorithms/shazam-it-music-processing-fingerprinting-and-recognition)

### **Libraries & Tools**

- **Torch + Torchaudio** (Feature extraction, LSTM training)
- **FAISS** (Fast similarity retrieval)
- **ONNX, PyTorch JIT** (Optimized inference)
- **SLURM + HPC Cluster** (Model training)
- **Weights & Biases (W&B)** (Experiment logging)

### **Infrastructure**

- **Dataset:** FMA (Full-Sized Songs)
- **Cluster:** **Nvidia A100 & V100 GPUs, 100Gb/s InfiniBand**