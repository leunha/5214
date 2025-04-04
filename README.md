# 5214
This project investigates the application of Rectified Flow, a novel generative modeling framework that uses Ordinary Differential Equations (ODEs) to transform data distributions along straightened trajectories, for medical image-to-image translation, specifically converting T1-weighted MRI scans to T2-weighted scans. This problem is compelling because different MRI modalities provide complementary diagnostic insights, yet acquiring multiple scans is costly and time-intensive. Automating the process could improve clinical efficiency and diagnostic precision, offering significant healthcare benefits. To acquire a deeper understanding of the problem, we will examine literature on image translation techniques like CycleGAN and Pix2Pix, as well as flow-based models, including the original Rectified Flow paper by Liu et al. The study will leverage the publicly available IXI dataset (https://brain-development.org/ixi-dataset/), containing paired T1 and T2 MRI scans, negating the need for new data collection. We propose implementing Rectified Flow with various neural network backbones (e.g., U-Net variants) to assess their impact on performance, modifying the original method by exploring the effects of reflow iterations and Euler discretization steps on generation quality and speed. Existing implementations from the original paper will be adapted, with enhancements focused on optimizing for 3D medical imaging demands. Evaluation will combine qualitative assessments—expecting clear, anatomically accurate T2 images visualized via plots—and quantitative metrics like Structural Similarity Index (SSIM) and Peak Signal-to-Noise Ratio (PSNR) to compare against baselines like GANs and diffusion models. Compared to these models, Rectified Flow's straight-path efficiency reduces computational overhead (potentially requiring just one Euler step post-reflow), while its stability avoids GANs' training instability and diffusion models' iterative slowness, promising high-fidelity, diverse outputs critical for medical accuracy. Therefore, we expect the proposed method to achieve better quality and efficiency than the traditional methods. Additionally, its domain transfer capability could extend to data augmentation for rare conditions, enhancing dataset robustness. Results will be statistically validated using paired t-tests to ensure significance, positioning Rectified Flow as a practical, high-quality solution for medical imaging tasks.

# Applying Rectified Flow for Medical Image Translation

## Problem Statement

This project investigates the application of Rectified Flow for medical image-to-image translation, specifically focusing on the task of transforming MRI images between different modalities (e.g., T1-weighted to T2-weighted scans). This is an interesting problem because:

1. **Clinical Relevance**: Different MRI modalities highlight different tissue properties, and having the ability to synthesize one modality from another can reduce the need for multiple scans, saving time and resources.
2. **Data Limitations**: Some modalities are more difficult to acquire or may be missing in historical datasets. The ability to generate these missing modalities can enhance retrospective studies.
3. **Theoretical Advancement**: Medical images represent complex high-dimensional distributions, providing a challenging real-world test for the Rectified Flow framework.

## Background and Context

The project builds upon several key areas of research:

1. **Rectified Flow** ([Liu et al., 2022](https://arxiv.org/abs/2209.03003)): A framework for learning ODEs that transfer data between distributions with straightened trajectories, enabling efficient one-step generation.
2. **Medical Image Translation**: Prior work using GANs (pix2pix, CycleGAN) and diffusion models for cross-modality synthesis.
3. **Neural ODEs**: The theoretical foundation for continuous-time generative models.

Key papers to review include:
- "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" (Liu et al., 2022)
- "Medical Image Synthesis with Context-Aware Generative Adversarial Networks" (Nie et al., 2017)
- "Unsupervised MR-to-CT Synthesis using Structure-Constrained CycleGAN" (Wolterink et al., 2017)
- "Multimodal Brain MRI Translation using Diffusion Models" (Recent work to compare against)

---

# Implementation Plan for T1 to T2 MRI Translation using Rectified Flow (2D Slices)

#

## Data Acquisition and Preprocessing


### Step 3: Load and Align Data Using TorchIO
- **Purpose**: Ensure T1 and T2 images are spatially aligned with consistent dimensions for accurate slice pairing.
- **Process**:
  1. Install TorchIO: `pip install torchio`.
  2. Load T1 and T2 images into a `SubjectsDataset`.
  3. Apply `Resample` to standardize spatial resolution and `CropOrPad` to uniform dimensions (e.g., 256x256x150 voxels).
- **Details**: Alignment ensures that corresponding T1 and T2 slices represent the same anatomical region.

### Step 4: Extract 2D Slices
- **Purpose**: Reduce memory usage by processing 2D slices instead of full 3D volumes, suitable for CPU-based training.
- **Process**:
  1. Extract slices from the middle region (e.g., 40%-60% of the z-axis) to capture significant brain structures.
  2. Save paired T1 and T2 slices as `.npy` files for efficient loading.
- **Details**: Uniform or anatomically informed slice selection ensures data relevance.

### Step 5: Normalize Slice Intensities
- **Purpose**: Standardize pixel intensities to enhance model training stability and performance.
- **Process**: Apply z-score normalization (subtract mean, divide by standard deviation) to each slice independently.
- **Details**: Independent normalization accounts for intensity variations within and across subjects.

---

## Addressing T1-T2 Alignment for One-to-One Correspondence

To ensure accurate T1-to-T2 translation, proper alignment between paired T1 and T2 scans is critical. This project addresses potential alignment issues through:

1. **Registration Pre-processing**: Before training, we apply an affine registration step using TorchIO's registration module to align T2 images to their corresponding T1 images, ensuring spatial correspondence.

2. **Visual Alignment Verification**: The `process_data.py` script generates alignment check visualizations showing T1, T2, and an overlay from multiple angles (axial, coronal, and sagittal) to verify spatial correspondence.

3. **Consistent Slice Extraction**: When creating the 2D dataset, slices are extracted from the same anatomical positions in both T1 and T2 volumes after registration.

4. **Overlay Visualization**: During training, results include overlay visualizations to help identify and address any remaining alignment issues.

Without proper alignment, the model would learn incorrect relationships between T1 and T2 appearances, resulting in poor translation quality. These alignment procedures ensure that the rectified flow model learns true one-to-one anatomical correspondence.

---

## Model Setup

### Step 6: Set Up MONAI U-Net for Rectified Flow
- **Purpose**: Utilize a robust U-Net architecture from MONAI as the backbone for the Rectified Flow model to estimate the velocity field.
- **Process**:
  1. Install MONAI: `pip install monai`.
  2. Define a `RectifiedFlowModel` class integrating the MONAI U-Net, adapting it to output velocity fields.
  3. Optionally freeze encoder layers to reduce training complexity on CPU.
- **Details**: The U-Net processes 2D slices, and Rectified Flow guides the transformation from T1 to T2.

---

## Training

### Step 7: Prepare the Dataset for Training
- **Purpose**: Organize 2D slices into a training-ready format.
- **Process**:
  1. Create a custom PyTorch dataset class to load paired T1 and T2 `.npy` files.
  2. Use a small batch size (e.g., 1 or 2) to fit CPU memory constraints.
- **Details**: Efficient data loading is critical for CPU-based training.

### Step 8: Train the Model
- **Purpose**: Train the Rectified Flow model to learn the T1-to-T2 mapping.
- **Process**:
  1. Adapt an existing Rectified Flow training script to the custom dataset and MONAI U-Net model.
  2. Configure hyperparameters for CPU (e.g., fewer epochs, smaller model size).
- **Details**: Training focuses on optimizing the velocity field estimation for slice translation.

---

## Evaluation

### Step 9: Evaluate the Model
- **Purpose**: Assess the quality of generated T2 slices against ground truth.
- **Process**:
  1. Compute metrics such as Structural Similarity Index (SSIM) and Peak Signal-to-Noise Ratio (PSNR).
  2. Visualize sample T1-to-T2 translations for qualitative evaluation.
- **Details**: Quantitative and qualitative assessments validate model performance.

---

## Potential Problems and Solutions

### 1. Data Size and Storage
- **Problem**: The full IXI dataset (10GB) and temporary tar files exceed typical storage capacities.
- **Details**: Extracting 50 subjects still requires temporarily handling 10GB of tar files.
- **Solution**: 
  - Extract only the required files using `tarfile` and delete tar files immediately after extraction.
  - Process in batches if disk space is limited.

### 2. Slice Selection
- **Problem**: Slices from the top or bottom of the volume may lack brain structures, reducing data utility.
- **Details**: Random or edge slices could lead to poor model training.
- **Solution**: 
  - Extract slices from the middle 40%-60% of the z-axis.
  - Optionally use anatomical landmarks (e.g., via TorchIO) to ensure brain presence.

### 3. Dimensionality Mismatch
- **Problem**: T1 and T2 images may differ in resolution or slice count, misaligning paired slices.
- **Details**: Inconsistent dimensions disrupt training data preparation.
- **Solution**: 
  - Use TorchIO’s `Resample` to align resolutions.
  - Apply `CropOrPad` to standardize volume dimensions before slicing.

### 4. Model Integration
- **Problem**: Integrating MONAI U-Net with Rectified Flow may require custom adjustments.
- **Details**: Rectified Flow expects velocity field outputs and time/condition inputs, differing from standard U-Net usage.
- **Solution**: 
  - Modify the U-Net output layer to produce velocity fields.
  - Ensure the model handles time and condition inputs as required by Rectified Flow.

### 5. Training Speed
- **Problem**: CPU-based training is significantly slower than GPU-based training.
- **Details**: Limited memory and processing power extend training duration.
- **Solution**: 
  - Reduce model size (e.g., fewer U-Net channels or layers).
  - Limit training to fewer epochs or a smaller slice subset.
  - Optimize data loading to minimize I/O bottlenecks.

### 6. Evaluation Metrics
- **Problem**: Low SSIM/PSNR scores may indicate insufficient training or model capacity.
- **Details**: Poor performance could stem from limited data or CPU constraints.
- **Solution**: 
  - Increase training epochs or dataset size if feasible.
  - Pre-train the U-Net as an autoencoder to improve feature extraction before Rectified Flow training.

---

