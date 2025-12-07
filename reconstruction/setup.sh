mkdir ~/workspace
cd ~/workspace
rm -rf ~/workspace/Project-Satanael
git clone https://github.com/corvus-rex/Project-Satanael.git
cd Project-Satanael/reconstruction/
git switch gan-retraining-pconv
conda env create -f environment.yml
conda init
conda activate inpainting
conda install -c conda-forge rclone -y
rm -rf experiments/
curl -L "https://drive.usercontent.google.com/download?id=1BwywcXg8KB8kRdmyMFseeHe8guEUIiZJ&confirm=xxx" -o experiments.tar
tar xvf experiments.tar
cd ~/workspace/Project-Satanael/reconstruction/
rclone config
rclone mount gdrive: ~/workspace/gdrive 
crontab -e
0 */4 * * * cp -r ~/workspace/Project-Satanael/reconstruction/experiments/aotgan_places2_pconv512 ~/workspace/gdrive/GAN_inpainting
python src/train.py --resume --tensorboard --early_stop=False --freeze_discriminator --freeze_generator=conf2