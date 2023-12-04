# Generating CT images based on magnetic resonance images
MR to CT translation has become very important in research in recent years. The importance of this
translation is that instead of irradiating harmful rays to the patient in order to obtain a real CT scan,
we can obtain a harmless MRI from the patient and convert it into a synthetic CT scan. In the past
years, various methods have been proposed to perform this translation, the latest of which is the use
of Deep Learning. Deep Learning methods presented in past researches are generally based on
convolutional neural networks and adversarial generative networks. In this project, diffusion models
have been used to translate MR to CT. Considering the superiority of diffusion models over
adversarial networks in the field of image generation, it was expected that this superiority would
also be seen in this project and we would be able to get better results than previous researches,
which happened. Five different methods of diffusion models were implemented and the results were
obtained on three datasets. Two methods, DDPM and PC, had the best performance among the five
methods, and their results were close to eachother in terms of the similarity criteria of two images.
If we consider the duration of image generation, we can say that DDPM method is the best. To
compare with previous works, the results of SSIM and PSNR were obtained on the Gold Atlas
dataset, which were obtained by DDPM method as 0.97 and 31.93, respectively. These results are
better than the best results of methods based on adversarial generative networks, which are 0.93 and
28.45. Also, the results on the CERMEP-IDB-MRXFDG dataset are also very good (SSIM 33.88,
PSNR 0.96), but due to lack of use in previous researches, it is not comparable.

In this project, I did an image to image translation task, specifically MRI to CT scan translation.

Reference: https://github.com/QingLyu0828/diffusion_mri_to_ct_conversion
