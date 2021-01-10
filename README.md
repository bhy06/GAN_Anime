# GAN_Anime
This is a PyTorch implementation of GANs, focusing on generating anime faces.

# To do:
- [x] Build anime-faces dataset
- [x] Implement GANs
- [ ] Implement StyleGANs
- [x] Implement Conditional GANs
- [ ] Import metric learning (triplet loss) 
 
# Anime-faces Dataset
All anime-faces images are collected and proprecessed by myself. Anime-style images of 45 tags are collected from [danbooru.donmai.us](https://danbooru.donmai.us/) using the crawler tool [gallery-dl](https://github.com/mikf/gallery-dl). The images are then processed by a anime face detector [lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface) in `build_animeface_dataset.py`. After cropping, meaningless images are deleted manually and the resulting dataset contains about 100,000 anime faces in total. For conditional GANs, anime-faces images of 20 tags (about 50,000 images) are utilized for training.

# Usage
To train the model (default=dcgan),
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

Training for 100 epochs (.gif) | Generated samples (.jpg) 
 -------- |-----------
![](https://github.com/bhy0v587/GAN_Anime/blob/main/resources/gif/cdcgan.gif) | ![](https://github.com/bhy0v587/GAN_Anime/blob/main/resources/image/cdcgan.jpg) 
 
## 7. ACGAN

Training for 100 epochs (.gif) | Generated samples (.jpg) 
 -------- |-----------
![](https://github.com/bhy0v587/GAN_Anime/blob/main/resources/gif/acgan.gif) | ![](https://github.com/bhy0v587/GAN_Anime/blob/main/resources/image/acgan.jpg) 
 

# Things I've learned

# Acknowledgements
- [jayleicn/animeGAN](https://github.com/jayleicn/animeGAN)
- [caogang/wgan-gp](https://github.com/caogang/wgan-gp)
- [igul222/improved_wgan_training](https://github.com/igul222/improved_wgan_training)
