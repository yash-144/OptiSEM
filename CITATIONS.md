# Citations and Attributions

In accordance with the competition guidelines allowing external data, and to provide full transparency on the architecture and tools used, we acknowledge the following datasets and papers:


## Architecture

1. **NAFNet (Nonlinear Activation Free Network)**
   - **Source:** [https://github.com/megvii-research/NAFNet](https://github.com/megvii-research/NAFNet)
   - **Citation:** Chen, L., Chu, X., Zhang, X., & Sun, J. (2022). Simple Baselines for Image Restoration. In *European Conference on Computer Vision (ECCV)*.
   - **Usage:** Our core feature extractor is built around the NAFBlock architecture, chosen for its industry-leading computational efficiency (removing costly softmax/GELU layers) which maximizes throughput on standard benchmark hardware (H100).
   - **License:** MIT License

## Evaluation Metrics

1. **LPIPS (Learned Perceptual Image Patch Similarity)**
   - **Source:** [https://github.com/richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity)
   - **Citation:** Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In *CVPR*.
   - **Usage:** Incorporated directly into our loss function (at 10% weighting), utilizing pre-trained AlexNet weights, to ensure our model does not collapse into blurry PSNR optimization, actively preserving perceptual structures.

## PyTorch Modules
- `pytorch_msssim`: Used for differentiable SSIM loss computation (MIT License).
