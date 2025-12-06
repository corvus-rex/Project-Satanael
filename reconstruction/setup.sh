mkdir ~/workspace
cd ~/workspace
rm -rf ~/workspace/Project-Satanael
git clone https://github.com/corvus-rex/Project-Satanael.git
cd Project-Satanael/reconstruction/
git switch gan-retraining-pconv
conda env create -f environment.yml
conda init
conda activate inpainting
rm -rf experiments/
curl -L "https://drive.usercontent.google.com/download?id=1BwywcXg8KB8kRdmyMFseeHe8guEUIiZJ&confirm=xxx" -o experiments.tar
tar xvf experiments.tar
python src/train.py --resume --tensorboard --early_stop=False --freeze_discriminator --freeze_generator=conf2