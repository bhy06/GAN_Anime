# GAN_Anime
This is a PyTorch implementation of GANs, focusing on generating anime faces.

# To do:
- [x] Build anime-faces dataset
- [x] Implement GANs
- [ ] Implement StyleGANs
- [x] Implement Conditional GANs
- [ ] Import metric learning (triplet loss) 
 
# Anime-faces Dataset
All anime-faces images are collected and proprecessed by myself. Anime-style images of 45 tags are collected from [danbooru.donmai.us](https://danbooru.donmai.us/) using the crawler tool [gallery-dl](https://github.com/mikf/gallery-dl). The images are then processed by a anime face detector [lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface). After cropping, meaningless images are deleted manually and the resulting dataset contains about 100,000 anime faces in total. For conditional GANs, anime-faces images of 20 tags are utilized for training.

# Run

# Model

# Results
## 1. GAN
## 2. DCGAN
## 3. W-DCGAN
## 4. W-DCGAN_GP
## 5. W-ResGAN_GP
## 6. CGAN
## 7. ACGAN

# Things I've learned

# Acknowledgements
- [jayleicn/animeGAN](https://github.com/jayleicn/animeGAN)
- [caogang/wgan-gp](https://github.com/caogang/wgan-gp)
- [igul222/improved_wgan_training](https://github.com/igul222/improved_wgan_training)
