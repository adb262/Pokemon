apt-get update && apt-get install -y python3-pip
pip install uv
uv sync
uv install -e .
source .venv/bin/activate
python train.py --frames-dir pokemon --use-s3 --batch-size 128 --num-epochs 3 --learning-rate 1e-4