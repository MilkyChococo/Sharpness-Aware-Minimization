# Sharpness-Aware-Minimization
Sharpness-Aware Minimization (SAM) was first introduced in the paper titled "Sharpness-Aware Minimization for Efficiently Improving Generalization" by Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur. Generally, SAM can optimize the function better than other optimizers in some situations by finding the noise on the training data, which makes the loss function max, plus parameters (weights) that help them adapt better to the changes in testing data.


Based on the knowledge from that paper, we continue learning about variants of SAM that can be applied to solving optimization problems: SSAMF, SSAMD, FriendlySAM, and IMBSAM. The source code is implemented in Python using the PyTorch framework and applied with a multivariable function.
