# sc_inpute_gan

In this work, we utilized a generative model to solve this technical issue by imputing dropouts in scRNA-seq data. We chose to base out project on an already developed method for imputing dropouts using GAN – termed scIGANs (Y. Xu et al., 2020). Here, we suggest two improvements for the method that will enable the generator to learn more realistic scRNA-seq datasets by receiving real data rather than normally distributed noise, while artificially imputing generated datasets during training. 

## Improvement 1: Simultaneous learning of biological gene expression and dropout 
Usually, generators are given random noise with some prior distribution as input. This enables the trained generator to form new data merely based on a noise. However, in imputation problems, our purpose is to correct existing, and available, data. As such, GAN trained for imputation could receive real data, rather than noise, as input. By feeding the generator the real data X as input, it will be able to learn the biological gene expression distribution along with the noise distribution at the same time. To train such GAN, we could apply artificial dropout on generated data, and then compute the loss as the ability of the discriminator to differentiate between the real data and the generated data that has been artificially dropped out.
	scIGANs utilizes the BEGAN method (Berthelot et al., 2017), in which the discriminator is an autoencoder, rather than a function that returns a value between 0 and 1. For realistic data, the input and output of the discriminator should be highly similar, while for unrealistic data (i.e., clearly fake), the difference between input and output of the discriminator should be high. As such, the loss function is computed as the absolute difference between the input and output of the discriminator.

Given the suggested modification, the objective of the used BEGAN is:

$L_D=L(x)-k_t L(Drop(G(x))$                                                          
$L_G=L(Drop(G(x))$                                            
$k_(t+1)=k_t+λ_k (γL(x)-L(Drop(G(x)))$                  
for each training step $t$

Where $G$ is a generator with parameters $θ_G$ and loss $L_G$, and $D$ is a discriminator with parameters $θ_D$ and loss $L_D$. 

Drop is the function that applies artificial dropouts based on fixed dropout parameters $θ_Drop$, based on which the probability of each entry in the generated gene expression matrix to be dropped out is computed. Specifically, $θ_Drop=(q,s)$ where q corresponds to the percentile value under which logit-transformed gene expression values are more likely to be dropped out, and s corresponds to the shape of the logit function. Accordingly, low gene expression values, with respective low logit-transformed values, are more likely to be dropped out, in accordance with the true nature of dropouts. Given a gene expression matrix $G(X)$, entry $G(X)_ij$ it is dropped out if a success is sampled from a distribution:

$Binom(n=1,p=\frac{1}{(1+e^(-s*(G(X)_ij'-percentile(G(X)',q)))})$

Where

$G(X)'=log⁡(G(X)+1)$

We note that while our loss is not optimized with respect to θ_Drop, the distribution of dropout is dependent on the gene expression values themselves, and as such, varies across datasets according to the range of gene expression values within it. Finally, the loss is defined as:

 
Where $γ$ is a hyperparameter within range [0,1] which equals $(Ε(L(Drop(G(x)))/(Ε(L(x)))$, $k_t$ is another hyperparameter within range [0,1] which corresponds to the tuning parameter of training step t, and it is learned with rate λ_k.
Notably, training the generator on the transformed images, which have generally higher dimension than noise sampled in a latent space, entails exceedingly long, and sometimes unfeasible, running times. We thus partition the scRNA-seq matrix to subsets of genes and apply the procedure on partitioned sub-datasets, transformed to images, rather than on the full data. To account for possible interactions between genes, we repeat random partitioning of the data and finally take the full imputed data to be the average of imputations across repeats. 
 

## Improvement 2: Architecture expansion for to accommodate for batch effect
scIGANs utilizes cell-type classification to enable learning of gene expression profiles catered to different cell types. The cell-type classification is given to the generator, and then concatenated to the convolved noise to obtain cell-type specific gene expression pattern. A similar approach could be utilized to account for batch effect. Specifically, rather than feeding the GAN cell data obtained using a single sequencing technique, we could integrate cell data from different sequencing techniques as one large training set. Using classification of each cell’s gene expression profile to its respective sequencing technique, we could apply a similar procedure to that applied on cell-type labels to account for sequencing-technique-specific noise (batch effect). Specifically, we will add an additional layer to the neural network of the generator (similar to the second layer of the generator described in the paper), which convolves sequencing-technique labels, sampled from a uniform distribution with the respective number of categories, and concatenates the output to the already concatenated noise and cell-type convolved labels.  The modification will encompass a generator defined as $G(x',L_(a_z ),L_(s_z );θ)$ where $x'$ is a transformed image per cell data on a subset of genes, $L_(a_z )~U(1,k)$ is the cell type label (given that the training data ranges across $k$ different cell types), and $L_(s_z )~U(1,m)$ is the sequencing technique label (with $m$ different sources for the scRNA-seq data integrated as the training data).


