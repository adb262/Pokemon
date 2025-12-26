# Pokemon IDM

The goal of this is to train a viable latent action IDM that understands the action space of pokemon. This IDM is completely self-supervised and is implemented with a VQVAE using the NSVQ paper. Ultimately, I am working on using to a flow matching diffusion transformer to generate the subsequent frames.

Because many pairs of frames are the same, I modified the reconstruction loss to focus more when the pixels change. This acts as a form of regularization and focus, preventing us from just reconstructing the original image.

## Data Collection

We can scrape a bunch of youtube videos with the following:
```
python -m idm.data_collection.pokemon_dataset_pipeline --max-videos 10 --scrape --clean --extract
```
Our end state is to have groups (starting with "pairs" i.e. group of 2) from which we can learn our latent actions. This means that we need to pair up frame x and frame x+1. Eventually, I will roll out support for larger groups, since the learning dynamics should be smoother.

## Training


### TODOS:
[] Flow Matching Transformer
[] Larger group dynamics (not just pairs)

Data
[X] Process parallelism
[] Loss is wrong
    - You shouldn't be using future frames, it's the current