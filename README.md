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

Starting with the Finite State Quantization (FSQ) for our tokenizer, we want to understand the impact of the quantization bins. Ideally, we are able to overfit on a small set first. So, we run two experiments:
- **Sending information through a straw**: We restrict our bins to be 2 2 2 2, meaning that we will lose a lot of information in our tokenization. Our vocab size is only 16 here.
- **Light Quantization**: Use bins 16 12 12 12, allowing vocabulary size of 36,864.

We expect to see an enormous delta, but it is much smaller than anticipated. So, let's try removing quantization entirely. Our encoder/decoder should just learn the identify function. 

Ablation results (see below):

![FSQ Ablation Results](public/fsq_ablation.png)


### TODOS:
[] Flow Matching Transformer
[] Larger group dynamics (not just pairs)
[] Do we really need video tokenization or would frame tokenization suffice?

Data
[X] Process parallelism
[X] Loss is wrong