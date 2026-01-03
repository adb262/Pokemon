# Pokemon IDM

The goal of this is to train a viable latent action IDM that understands the action space of pokemon. This IDM is completely self-supervised and is implemented with a VQVAE using the NSVQ paper. Ultimately, I am working on using to a flow matching diffusion transformer to generate the subsequent frames.

Because many pairs of frames are the same, I modified the reconstruction loss to focus more when the pixels change. This acts as a form of regularization and focus, preventing us from just reconstructing the original image.

## Data Collection

We can scrape a bunch of youtube videos with the following:
```
python -m src.data.scraping.pokemon_dataset_pipeline --clean --extract --summary --extract --jump_seconds 5.0 --num_video_workers 8 --num_upload_threads 16 --use_s3
```
Our end state is to have groups (starting with "pairs" i.e. group of 2) from which we can learn our latent actions. This means that we need to pair up frame x and frame x+1. Eventually, I will roll out support for larger groups, since the learning dynamics should be smoother.

## Training

Starting with the Finite State Quantization (FSQ) for our tokenizer, we want to understand the impact of the quantization bins. Ideally, we are able to overfit on a small set first. So, we run two experiments:
- **Sending information through a straw**: We restrict our bins to be 2 2 2 2, meaning that we will lose a lot of information in our tokenization. Our vocab size is only 16 here.
- **Light Quantization**: Use bins 16 12 12 12, allowing vocabulary size of 36,864.

We expect to see an enormous delta, but it is much smaller than anticipated. So, let's try removing quantization entirely. Our encoder/decoder should just learn the identify function. 

Ablation results (see below):

![FSQ Ablation Results](public/fsq_ablation.png)

## Training Video Tokenizer
Local training
```python -m scripts.video_tokenizer.train --frames_dir pokemon_frames/pokemon_emerald --num_images_in_video 5 --batch_size 2  --save_dir fsq_full_train --checkpoint_dir full_train --bins 8 8 6 5 --local_cache_dir ''```

S3 
```
python -m scripts.video_tokenizer.train --frames_dir pokemon_frames/pokemon_emerald --num_images_in_video 5 --batch_size 2  --save_dir fsq_full_train --checkpoint_dir full_train --bins 8 8 6 5 --use_s3 true
```

### TODOS:
[] Flow Matching Transformer
[] Larger group dynamics (not just pairs)
[] Do we really need video tokenization or would frame tokenization suffice?

Data
[X] Process parallelism
[X] Loss is wrong
[] Koga gym is known poisonous bc of teleportation
[] Horrible Heart Gold/Soul Silver
[] When we enter into trainer battle
[] We cannot tell when the frame is exactly the same


## TODO
X Use additive embeddings
X Move to single action per frame
- When char is moving, everything should be moving. Filter to frames where the redisual is every frame or nothing
- Use EMA codebook updates
- Just look at center of the frames to determine action
    - Can't bc some frames are un-croppsed
- Dealing with codebook collapse now. Tried resetting but just collapses elsewhere. Might want to just focus on the center frame (only care about the char)
- More homogeneous data
- Emergence of no-action quantization. Successfully classifying when there is no action


Create streamlit for labeling images as valid or invalid. Highlight spans over a low calibre preview of an entire video

This streamlit should load in a .mp4. It should show the video preview in a sort of spread out "timeline". This timeline should be very low resolution so to allow us to view large chunks of the video at once. The video should not play, but rather it should just show the expanded frames. We should show 5 frames/second. 
The user interacts with this by dragging intervals over the spans of frames that we want to label as "valid". When we save the annotation, the frame indices for the valid frames are written to a .json in the directory (create if doesn't exist) f"labeled_frames/{video_name_without.mp4}.json". The structure of the json is a list of lists, where each list represents a chunk of viable frames.

Some thoughts:
- Because the videos can be very long, we likely want to stream chunks of it in. We will not be able to show every single frame at once, and should just show windows. We need to be able to label large chunks of the video at once so that it is simple to go through many very large videos.


## Data Scripts

```
python -m scripts.data.sync_dataset_to_s3 --source pokemon_frames/pokemon_emerald_train_0_9_5_frames.json --bucket [bucket_name] --verbose
```

## Observations

### Latent Action Model
During training, the easiest way to minimize the loss early is to just regurgitate the previous frame. Over time, this transitions to learning a sort of middle ground between the previous and next frame. This manifests as somewhat of a motion blur, where you see the previous frame and the current frame together. Below is an example of this phenomena. This is from epoch 50 over a dataset of only 100 frame transitions

![FSQ Ablation Results](public/lam_frame_blurring.png)

Eventually we overfit and learn the exact next frame. I am working on experiments of this at scale.