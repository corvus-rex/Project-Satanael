## AOT-GAN training setup
This guide details concise step-by-step process on how to run a training session for AOT-GAN 
including installing relevant dependencies, downloading training dataset and their mask, and 
setting up an `rclone` upload pipeline to periodically save training results (Tensorboard 
logs and checkpoint file) to a remote Gdrive directory. This documentation describes the bash
file `setup.sh`, it is recommended however for first time setup to run each command separately
and sequentially as outlined in the guideline below.

### Setting up workspace

1. `mkdir ~/workspace`
2. `cd ~/workspace`
3. `rm -rf ~/workspace/Project-Satanael`
4. `git clone https://github.com/corvus-rex/Project-Satanael.git`
Clone the adversarial defense repository 
5. `cd Project-Satanael/reconstruction/`
6. `git switch gan-retraining-pconv`
Switch git branch to `gan-retraining-pconv`, this branch will be used for development of future
training-related changes

### Installing dependencies

7. `conda env create -f environment.yml`
Anaconda has already been pre-installed in the HPC, use `environment.yml` to install Python dep-
endencies
8. `conda init`
9. `conda activate inpainting`
10. `conda install -c conda-forge rclone -y`
Install `rclone`, used to sync training result to a Google Drive directory

### Downloading training dataset
11. `rm -rf experiments/`
Delete previously-cloned experiments dir to replace with a new one pre-configured with the dataset
12. `curl -L "https://drive.usercontent.google.com/download?id=1BwywcXg8KB8kRdmyMFseeHe8guEUIiZJ&confirm=xxx" -o experiments.tar`
13. `tar xvf experiments.tar`
14. `cd ~/workspace/Project-Satanael/reconstruction/`

### Configure `rclone` (Optional, but needs to be done sooner or later)
15. `rclone config`
Configure rclone, because the HPC is a headless environment, a seperate machine (Adrian's local machine) is required to
obtain the Gdrive's Oauth key. When prompted,
```
Use web browser to automatically authenticate rclone with remote?
 * Say Y if the machine running rclone has a web browser you can use
 * Say N if running rclone on a (remote) machine without web browser access
```
press N, an authentication token in the form of URL will be provided, in exchange the config will ask for authentication
key. Send this URL to Adrian and an Oauth key will be provided in return which allows config process to resume.
For more info regarding `rclone`'s setup: https://rclone.org/drive/ 
16. `rclone mount --daemon gdrive: ~/workspace/gdrive` 
17. `crontab -e`
Add the following crontab job to regularly sync training checkpoint and Tensorboard log
```
0 */4 * * * cp -r ~/workspace/Project-Satanael/reconstruction/experiments/aotgan_places2_pconv512 ~/workspace/gdrive/GAN_inpainting
```

### Initialize training
18. `python src/train.py --resume --tensorboard --early_stop=False --freeze_discriminator --freeze_generator=conf2`
What each flag does:

- `--resume` loads the latest checkpoint (zeroth iteration in the dataset)
- `--tensorboard` enables live logging
- `--freeze_discriminator` freezes discriminator in training
- `--freeze_generator=conf2` applies partial generator freezing (freeze all but decoder layers)
- `--early_stop=False` disables early stopping logic