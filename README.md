# TASK-3 NEURAL STYLE TRANSFER
COMPANY: CODTECH IT SOLUTIONS

NAME: Priyanshu V. Patel

INTERN ID: C0DF191

DOMAIN: Artificial Intelligence Markup Language.

DURATION: 4 WEEEKS

MENTOR: NEELA SANTOSH

# Neural Style Transfer
This code implements a Neural Style Transfer (NST) algorithm using the pretrained VGG-19 model in PyTorch. NST blends the content of one image with the style of another. The goal is to optimize a target image (initially the content image) by minimizing a combined loss function that incorporates both content and style losses.

The code begins by importing necessary libraries like torch for deep learning operations, torchvision.transforms for image preprocessing, PIL for image manipulation, and matplotlib.pyplot and numpy for plotting and numerical operations. The device is set to use CUDA if a GPU is available, otherwise, it defaults to the CPU for computations. word count karna toh The VGG-19 model is loaded from PyTorch’s model library in feature extraction mode. All the model parameters are frozen (no gradients are calculated for them). This model will be used to extract features from different layers, as they capture varying levels of abstraction in images.

The model_activations function runs an image through the layers of the VGG-19 model and stores the activations (feature maps) from specific layers. These layers are selected because they best capture the content and style characteristics of the image. The content and style images are loaded, resized, and normalized to fit the model’s input requirements.

To visualize the images, the imcnvt function converts the processed PyTorch tensor back to a displayable format using matplotlib. It also unnormalizes the image for better visual interpretation.

A critical part of the style transfer process is the computation of the Gram matrix. This matrix captures correlations between feature maps and is essential for calculating style loss. The gram_matrix function computes this for the style image and the target image.

The loss function is a combination of two components: content loss and style loss. The content loss is calculated as the mean squared error (MSE) between the content features of the target and the content image. The style loss is based on the difference between the Gram matrices of the style and target images. Each layer in the model has a weight (specified by style_wt_meas) that controls its contribution to the total style loss.

The target image is then optimized by minimizing the combined loss function. The optimizer used is Adam, and the target image is updated iteratively over 100 epochs. Every 10 epochs, the total loss is printed, and every 100 epochs, the target image is visualized and saved.

This model can be used in a variety of fields. In art and design, it allows artists to create unique artwork by combining the content of one image with the artistic style of another. In image and video processing, it can stylize videos or stills, providing new forms of visual expression. In marketing and social media, businesses and influencers can use it to create eye-catching visuals for advertisements or online content, blending a brand’s image with popular artistic styles. For personalized content creation, this model can generate custom wallpapers, greeting cards, or prints by mixing personal photos with a chosen style. It also finds applications in augmented reality, where it can enhance virtual elements by applying artistic styles that blend seamlessly with the real-world environment.

The tools and resources used to implement this model include YouTube for tutorials on neural style transfer and the use of VGG-19, ChatGPT for clarifying concepts related to deep learning, GeeksforGeeks for understanding convolutional networks and style transfer, DeepSeek for references on efficient PyTorch implementations, advanced Python knowledge for utilizing PyTorch and tensor operations, and VS Code as the integrated development environment for writing and running the code.

This combination of tools and resources makes it possible to efficiently implement neural style transfer and apply it to a wide range of creative and professional use cases.

# OUTPUT
