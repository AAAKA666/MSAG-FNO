# A Multi-Scale Adaptive Gated Fourier Neural Operator (MSAG-FNO) for Urban Flood Prediction
##  MSAG-FNO
![alt text](MSAG-FNO/MSAG-FNO
/figures/Figure2.png)
MSAG-FNO model architecture. The detailed structure of a single MSAG-FNO layer is presented, including three parallel spectral convolution branches with different frequency truncations (K₀=16, K₁=8, K₂=4). An adaptive gated MLP dynamically generates scale weights α₀, α₁, α₂ based on the input features, performs a weighted sum Σ, and then executes the inverse Fourier transform F⁻¹.

##  Transfer Learning-based MSAG-FNO
![alt text](MSAG-FNO/MSAG-FNO
/figures/Figure3.png)
To enhance the transferability and generalization capability of urban flood forecasting across different scenarios, this study proposes a fine-tuned MSAG-FNO for effective prediction in target domains. The fine-tuned MSAG-FNO consists of two stages: pre-training on the source domain and fine-tuning on the target domain. During the source domain pre-training stage, the model learns a generalizable physical representation of urban hydrodynamics. In the target domain fine-tuning stage, the pre-trained weights are transferred to the new target domain, and the input data is appropriately adjusted to align with the parameters of the target domain.



、
