# GAN_Anime
This is a PyTorch implementation of GANs, focusing on generating anime faces.

# To do:
- [x] Build anime-faces dataset
- [x] Implement GANs
- [ ] Implement StyleGANs
- [x] Implement Conditional GANs
 
# Anime-faces Dataset
All anime-faces images are collected and proprecessed by myself. Anime-style images of 45 tags (`tags.txt`) are collected from [danbooru.donmai.us](https://danbooru.donmai.us/) using the crawler tool [gallery-dl](https://github.com/mikf/gallery-dl). After deleting unrelated images without anime-faces, the images are then processed by a anime face detector [lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface) in `build_animeface_dataset.py`. After cropping, meaningless images are deleted manually and the resulting dataset contains about 100,000 anime faces in total. For conditional GANs, anime-faces images of 20 tags (`tags_20.txt`) (about 50,000 images) are utilized for training.

Dataset is available [here](https://drive.google.com/file/d/1aHmdEOHii2qDBFjUmHOhClVYmQCPKEJd/view?usp=sharing).

# Usage
To train the model (default：dcgan),
```
python train.py --dataRoot path_to_dataset --cuda
```

# Models
In `train.py` multiple gans are available by initializing `--model`:
- GAN: use `--model 0` to run `models/gan.py`
- DCGAN: use `--model 1` to run `models/dcgan.py`
- W-DCGAN: use `--model 2` to run `models/wdcgan.py`
- W-DCGAN_GP: use `--model 3` to run `models/wdcgan_gp.py`
- W-ResGAN_GP: use `--model 4` to run `models/wresgan_gp.py`
- CGAN: use `--model 5` to run `models/cdcgan.py`
- ACGAN: use `--model 6` to run `models/acgan_resnet.py`

# Results
## 1. GAN

Training for 100 epochs (.gif) | Generated samples (.jpg) 
 -------- |-----------
![](https://github.com/bhy0v587/GAN_Anime/blob/main/resources/gif/gan.gif) | ![](https://github.com/bhy0v587/GAN_Anime/blob/main/resources/image/gan.jpg) 
 
## 2. DCGAN

Training for 100 epochs (.gif) | Generated samples (.jpg) 
 -------- |-----------
![](https://github.com/bhy0v587/GAN_Anime/blob/main/resources/gif/dcgan.gif) | ![](https://github.com/bhy0v587/GAN_Anime/blob/main/resources/image/dcgan.jpg) 
 
## 3. W-DCGAN

Training for 100 epochs (.gif) | Generated samples (.jpg) 
 -------- |-----------
![](https://github.com/bhy0v587/GAN_Anime/blob/main/resources/gif/wdcgan.gif) | ![](https://github.com/bhy0v587/GAN_Anime/blob/main/resources/image/wdcgan.jpg) 
 
## 4. W-DCGAN_GP

Training for 100 epochs (.gif) | Generated samples (.jpg) 
 -------- |-----------
![](https://github.com/bhy0v587/GAN_Anime/blob/main/resources/gif/wdcgan_gp.gif) | ![](https://github.com/bhy0v587/GAN_Anime/blob/main/resources/image/wdcgan_gp.jpg) 
 
## 5. W-ResGAN_GP

Training for 100 epochs (.gif) | Generated samples (.jpg) 
 -------- |-----------
![](https://github.com/bhy0v587/GAN_Anime/blob/main/resources/gif/wresgan_gp.gif) | ![](https://github.com/bhy0v587/GAN_Anime/blob/main/resources/image/wresgan_gp.jpg) 
 
## 6. CGAN

Generated samples are based on the following category order, where the images of each category are shown in each row.

From top to bottom: green_hair, orange_hair, purple_hair, silver_hair, blue_eyes, green_eyes, pink_eyes, red_eyes

Training for 100 epochs (.gif) | Generated samples (.jpg) 
 -------- |-----------
![](https://github.com/bhy0v587/GAN_Anime/blob/main/resources/gif/cdcgan.gif) | ![](https://github.com/bhy0v587/GAN_Anime/blob/main/resources/image/cdcgan.jpg) 
 
## 7. ACGAN

Generated samples are based on the following category order, where the images of each category are shown in each row.

From top to bottom: green_hair, orange_hair, purple_hair, silver_hair, blue_eyes, green_eyes, pink_eyes, red_eyes

Training for 100 epochs (.gif) | Generated samples (.jpg) 
 -------- |-----------
![](https://github.com/bhy0v587/GAN_Anime/blob/main/resources/gif/acgan.gif) | ![](https://github.com/bhy0v587/GAN_Anime/blob/main/resources/image/acgan.jpg) 
 

# Things I've learned
- GAN is really hard to train since it is difficult to balance D and G.
- DCGAN generally works better than GAN and it can generate clearer images with details.
- WGAN trains more stably and has the metric to show the convergence during training, also avoids mode collapse problem.
- WGAN-GP using gradient penalty shows more powerful performance and better generated images than WGAN using weight clipping.
- CGAN is also hard to train and easily causes mode collapse problem.
- ACGAN seems more stable and powerful in generating conditional images.

# Tips based on personal experience
- The most important thing for training GANs is to learn to balance D and G.
- Add noise to D's inputs and labels helps stablize training.
- Adam is always good, but exponetially decaying learning rate seems not so helpful and makes no significant differences.
- Training D several times than G sometimes seems helpful (WGAN) but easily makes D so strong thus upsets the existing balance.
- Giving D higher learning rate than G seems lead to better results.
- D should be a little more powerful to lead G to generate better images.

# Acknowledgements
- [jayleicn/animeGAN](https://github.com/jayleicn/animeGAN)
- [caogang/wgan-gp](https://github.com/caogang/wgan-gp)
- [igul222/improved_wgan_training](https://github.com/igul222/improved_wgan_training)
- [clvrai/ACGAN-PyTorch](https://github.com/clvrai/ACGAN-PyTorch)
