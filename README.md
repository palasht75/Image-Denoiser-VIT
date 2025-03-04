# Image Denoiser using Vision Transformer (ViT)

This project implements a denoising autoencoder based on Vision Transformers (ViT) to remove noise from colon histopathological images. The model is trained using a combined loss function (MSE + SSIM loss) and supports live visualization of training progress through custom callbacks. The dataset used is from Kaggle: [Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images).

---

## Methodology

The denoising system is built around a **Vision Transformer Denoiser** with the following key components:

- **Initial Convolution Layer:**  
  Extracts local features from the input image. This early extraction is crucial for preserving fine details in medical images.

- **Patch Embedding:**  
  The feature map is divided into small patches (set by `PATCH_SIZE = 8`), which are then projected into an embedding space (`EMBED_DIM = 128`) and enriched with positional encodings. A smaller patch size allows the model to capture finer details, which is essential for the subtle differences in histopathological images.

- **Transformer Blocks:**  
  Multiple transformer blocks (with a depth of `DEPTH = 12`) using multi-head self-attention (`NUM_HEADS = 8`) and an MLP of dimension (`MLP_DIM = 256`) process the patch embeddings. These blocks allow the model to capture long-range dependencies and complex relationships across the image.

- **Decoder with Skip Connections:**  
  After processing, the model reshapes the output back into a spatial grid and upsamples it using transposed convolutions. Skip connections from the initial convolution layer help recover and retain fine-grained details that might otherwise be lost during deep feature extraction.

- **Combined Loss Function:**  
  The loss function is a weighted sum of **MSE loss** and **SSIM loss**. MSE ensures that the denoised image's pixel values closely match those of the clean image, while SSIM ensures that the structural similarity is maintained. This combination is vital for medical imaging tasks where both numerical accuracy and perceptual quality are important.

---

## Parameter Choices

- **`IMG_SIZE = 64`:**  
  A balance between computational efficiency and retaining essential image details. Resizing images to 64×64 helps reduce the model complexity while preserving critical features.

- **`PATCH_SIZE = 8`:**  
  A smaller patch size is used to ensure that the model captures detailed local patterns, which are particularly important in medical images.

- **`EMBED_DIM = 128`:**  
  The dimension of the patch embeddings, providing a rich representation for each patch.

- **`NUM_HEADS = 8`:**  
  Multi-head attention enables the model to attend to different parts of the image simultaneously, which enhances its ability to capture global context.

- **`MLP_DIM = 256`:**  
  Determines the capacity of the feed-forward network within the transformer blocks, allowing the model to learn complex, non-linear relationships.

- **`DEPTH = 12`:**  
  A deeper transformer network increases the model's ability to learn from complex patterns in the image data.

- **Standard Hyperparameters:**  
  Parameters like `BATCH_SIZE`, `EPOCHS`, `NOISE_STDDEV`, and `LEARNING_RATE` are set based on empirical validation to ensure efficient training and robust performance.

---

## Live Training Visualization Callbacks

To monitor training progress in real-time, two custom callbacks are implemented:

### 1. LivePlotCallbackBatch
- **Description:**  
  This callback updates the visualization after every training batch. It displays a row with three images: the clean image, the noisy input, and the denoised output from the model.
- **Usage:**  
  Use this callback when detailed, step-by-step visualization is needed—ideal for debugging or when working on smaller datasets. However, frequent updates may slow down training due to blocking operations.

### 2. LivePlotCallbackAsync
- **Description:**  
  This callback updates the visualization asynchronously every few batches (e.g., every 10 batches). It utilizes Python's `threading` module to offload plotting to a separate thread.
- **Significance of Threading:**  
  Threading allows the plotting to occur without blocking the main training loop. This ensures that while the model continues to train efficiently, the visual feedback is updated in the background.
- **Usage:**  
  Recommended for regular training on larger datasets where minimizing training interruption is crucial. It offers a good balance between live monitoring and efficient GPU utilization.

**When to Use Which Callback:**
- **Use `LivePlotCallbackBatch`** when you require granular, per-batch updates (e.g., for intensive debugging or detailed analysis on a smaller subset of data).
- **Use `LivePlotCallbackAsync`** for regular training sessions where maintaining high training speed is important, yet you still desire periodic visual feedback.

---

## Project Summary

This project demonstrates a sophisticated approach to image denoising using Vision Transformers, integrating advanced methodologies like patch embedding, multi-head self-attention, and combined loss functions. The inclusion of live training visualization—implemented through both blocking and asynchronous callbacks—provides immediate insights into model performance, enabling quick adjustments and optimizations.

For further details, please refer to the accompanying Jupyter Notebook, which contains the complete code, implementation details, and real-time training visualization.

For any questions or suggestions, please contact the project maintainer.