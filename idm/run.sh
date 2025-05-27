apt-get update && apt-get install -y python3-pip
pip install uv
uv sync
source .venv/bin/activate
uv pip install -e .

wandb login $WANDB_API_KEY

python train.py --frames_dir pokemon --use_s3 true --batch_size 16 --num_epochs 3 --learning_rate 1e-4 --experiment_name test_s3 --seed_cache