# Sharpness-Aware-Minimization
Optimizing model parameters remains a challenging problem for researchers, as it plays a crucial role in enhancing the performance of generative models. Modern deep learning models are often highly overparameterized, capable of memorizing the entire training set. This leads to the problem of overfitting, reducing the model’s generalization capability for unseen data.

On 3 Oct 2020, Sharpness-Aware Minimization (SAM) was first introduced in the paper titled "Sharpness-Aware Minimization for Efficiently Improving Generalization" by Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur. Generally, SAM can optimize the function better than other optimizers in some situations by finding the noise on the training data, which makes the loss function max, plus parameters (weights) that help them adapt better to the changes in testing data. SAM aims to reduce the sharpness of minima to ensure the model is more stable on new data. 

![Project Logo](figure1.png). 

Sharpness indicates the model's sensitivity to small changes in its parameters (weights). There are two types of Sharpness: Sharp Minimum and Flat Minimum
Based on the knowledge from that paper, we continue learning about variants of SAM that can be applied to solving optimization problems: SSAMF, SSAMD, FriendlySAM, and IMBSAM. The source code is implemented in Python using the PyTorch framework and applied with a multivariable function.
