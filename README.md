# Pokemon IDM

The goal of this is to train a viable latent action IDM that understands the action space of pokemon. This IDM is completely self-supervised and is implemented with a VQVAE using the NSVQ paper. Ultimately, I am working on using to a flow matching diffusion transformer to generate the subsequent frames.

Because many pairs of frames are the same, I modified the reconstruction loss to focus more when the pixels change. This acts as a form of regularization and focus, preventing us from just reconstructing the original image.

## Data Collection

We can scrape a bunch of youtube videos with the following:
```
python -m src.data.scraping.pokemon_dataset_pipeline --scrape --clean --extract --summary --jump_seconds 5.0 --num_video_workers 8 --num_upload_threads 16 --use_s3
```

To download from a specific YouTube video or playlist URL:
```
python -m src.data.scraping.pokemon_dataset_pipeline --scrape --clean --extract --summary --video_url "https://www.youtube.com/playlist?list=PL7RjQqHgsQeQsxpfp_rPZJR-44YATOPZ5" --game_name "Pokemon Emerald" --jump_seconds 5.0 --num_video_workers 8 --num_upload_threads 16
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

### TODOS

- [ ] Flow Matching Transformer
- [ ] Larger group dynamics (not just pairs)
- [ ] Do we really need video tokenization or would frame tokenization suffice?
- [X] Use additive embeddings
- [X] Move to single action per frame
- [ ] When char is moving, everything should be moving. Filter to frames where the residual is every frame or nothing
- [ ] Use EMA codebook updates
- [ ] Just look at center of the frames to determine action
    - [ ] Can't because some frames are un-cropped
- [ ] Deal with codebook collapse (tried resetting but just collapses elsewhere; might want to focus on center frame, only care about the char)
- [ ] Use more homogeneous data
- [ ] Handle emergence of no-action quantization; successfully classify when there is no action
- [ ] Test out JEPA style learning
- [ ] Test out DiT
- [ ] Test out BPTT using Gumbel Softmax for iterative decoding (bridge train/inference gap)
- [ ] Fix data leakage between train/test (the test set can be interspersed with segments from training)
- [ ] Trim out the borders present

#### Data

- [x] Process parallelism
- [x] Loss is wrong
- [ ] Koga gym is known poisonous bc of teleportation
- [ ] Horrible Heart Gold/Soul Silver
- [ ] When we enter into trainer battle
- [ ] We cannot tell when the frame is exactly the same


## Data Scripts

```
python -m scripts.data.sync_dataset_to_s3 --source pokemon_frames/pokemon_emerald_train_0_9_5_frames.json --bucket [bucket_name] --verbose
```

## Observations

### Latent Action Model
During training, the easiest way to minimize the loss early is to just regurgitate the previous frame. Over time, this transitions to learning a sort of middle ground between the previous and next frame. This manifests as somewhat of a motion blur, where you see the previous frame and the current frame together. Below is an example of this phenomena. This is from epoch 50 over a dataset of only 100 frame transitions

![FSQ Ablation Results](public/lam_frame_blurring_epoch_50.png)

Eventually we overfit and learn the exact next frame. I am working on experiments of this at scale.


### Video Tokenizer

We first train a 12 million parameter tokenizer. We select Finite Scalar Quantization (Mentzer, 2023) because of the simplicity and the empirical lack of quality degradation when scaling vocab size. This tokenizer is trained with 2 heads, 2 encoder layers, 2 decoder layers, d_model = 256, image shape = 260x260, patch_size = 8, and a vocab size of 1,920. Our results are _decent_, but we certainly are leaving a lot on the table. 
![2k Vocab Results](public/tokenizer_2k_vocab_260_pixel_8_patch.png)

In the genie papers, they train a 200 million parameter encoder with vocab size of 1k. Their tokenizer views images as 160x90, using a patch size of 4. It is worthwhile for us to run some ablations over the different image qualities and vocab sizes.
![Tokenizer Shape Ablation](public/tokenizer_scaling_image_size_impact_eval_loss.png)
![Tokenizer Shape Ablation Frechet](public/tokenizer_scaling_image_size_impact_frechet.png)

As expected, increasing vocabulary size improves performance. However, this also creates a more difficult learning objective downstream. We see that the largest delta actually comes from scaling the image quality. There is a ~20% improvement in frechet distance simply by dropping from 260x260 with a patch size of 8 to 128x128 with a patch size of 4. As a result, we opt to use this moving forward.
![2k Vocab Results with 128x128/patch size of 4](public/tokenizer_2k_vocab_128_pixel_4_patch.png)

These results are empirically a bit better, but they still don't pass the eye test. We want to feel like we're actually in the game, and the video tokenizer decoder is integral to the reconstruction performance of our dynamics model downstream. So, we'd like to see the impact of scaling the tokenizer itself. We opt for a 50m and 100m model by scaling d_model to 512 and using num heads = 8, setting encoder/decoder layers to 4 and 8 respectively. 
![Tokenizer Scaling Ablation](public/tokenizer_scaling_params_placeholder.png)

Something is clearly not right with our 100m tokenizer... Let's take a look
![Tokenizer 100m Quality Issues](public/tokenizer_100m_quality_issues.png)

It seems that we converge on some local minima. Let's take a look at our gradients:
![Tokenizer 100m Gradients](public/tokenizer_100m_gradient_sparsity.png)
The gradients are only active during the first ~5k steps. Since we're using gradient accumulation of 16, This means we really only take 300 steps before converging to our local minima. Let's try adding some better weight initialization, dropping our learning rate, and adding a warmup.

I'm reasonably happy with the 50m model performance and it comes at a pretty cheap latency hit. Here are some eval samples:
![Tokenizer 50m quality](public/tokenizer_50m_quality.png)

Genie explores scaling the decoder separately. Since the decoder is essentially the bottleneck for reconstruction fidelity, this is worthwhile to pursue. I may come back to this.

- 12M Tokenizer: https://wandb.ai/adb262-cornell-university/pokemon-vqvae/runs/lhek7ncp
- 50M Tokenizer: https://wandb.ai/adb262-cornell-university/pokemon-vqvae/runs/mu0wmzo3
- 100M Tokenizer: https://wandb.ai/adb262-cornell-university/pokemon-vqvae/runs/t0e7zv8b


**You'll notice that a lot of the frames have not been trimmed. I originally removed these artifacts but decided that they actually may improve diversity of generation over time.

- The lookup table only looks at users current action...
- Train over variable lengths


## TODOs
- [X]: Update data loader to create an n length sequence of frames, where n >> num_images_in_video, split this up into k samples where each len(sample) == num_images_in_video, k = n - (num_images_in_video - 1). This is because each sequence must be of length num_images_in_video, so we cannot use the first (num_images_in_video - 1) frames as training data. Importantly, we must shuffle our dataset post collection, such that we have a reasonable data mixture for our dynamics model.
- []: Use data from https://www.youtube.com/playlist?list=PL7RjQqHgsQeQsxpfp_rPZJR-44YATOPZ5
- [X]: Update our MaskGiT implementationm
- []: Switch to Zarr arrays


## From Pokemon to Pong
Pokemon has a lot of issues. For one, it's mostly an animated game. There are substantially more frames spent with inaction than there are in open world navigation. For our basic repro, we want the simplest possible environment. We'd love to have just the sprite running around but our data is flooded with cut scenes and dialogue. Initial attempts at algorithmic filtering were crushed by edge cases and variants. The next step would be: either spend money to have the frames labeled by a VLLM (navigable/not navigable), OR train a model on a small set of data. This is extremely feasible, but I'm interested in proving that the pipeline works first. So, Pong is a natural, simple environment. There are still cases of inactivity here, but filtering is quite a bit easier.

## Pong Experiments
Tried weighting based on white pixelation but that collapsed quickly. We were able to achieve near perfect reconstruction with the following checkpoint.

Pong tokenizer
```
CUDA_VISIBLE_DEVICES=7 python -m scripts.video_tokenizer.train \
  --dataset_type atari_pong \
  --atari_pong_data_dir data/atari_pong \
  --atari_pong_require_full_gameplay \
  --image_size 84 \
  --patch_size 4 \
  --num_images_in_video 5 \
  --batch_size 32 \
  --num_epochs 3 \
  --dataset_limit 1_000_000 \
  --bins 8 8 6 5 \
  --save_dir fsq_tokenizer_atari_pong_large_ds_clipped \
  --checkpoint_dir fsq_tokenizer_atari_pong_large_ds_clipped \
  --logging-backend tensorboard \
  --tensorboard_dir tokenizer_runs \
  --experiment_name fsq_tokenizer_atari_pong_5_frames_large_ds_clipped \
  --reconstruction_loss_type clipped_l2 --l2_clip_c 10.0 \
  --early_stopping_patience 10 \
  --early_stopping_min_delta 1e-5 \
  > fsq_tokenizer_atari_pong_ds_clipped.log 2>&1
```

TODO:
- []: Rerun ablations with new upsamplers

![Pong Tokenizer Results](public/pong_tokenizer_50m_comparison_grid.png)
Run here: https://wandb.ai/adb262-cornell-university/pokemon-vqvae/runs/tsznmr1u?nw=nwuseradb262 (fsq_tokenizer_atari_pong/checkpoint_epoch1_batch3000.pt)
```
CUDA_VISIBLE_DEVICES=5 python -m scripts.dynamics_model.train \
  --dataset_type atari_pong \
  --atari_pong_data_dir data/atari_pong \
  --tokenizer_checkpoint_path fsq_tokenizer_atari_pong/checkpoint_epoch1_batch3000.pt \
  --image_size 128 \
  --patch_size 4 \
  --num_images_in_video 5 \
  --batch_size 4 \
  --save_dir dynamics_model_atari_pong \
  --checkpoint_dir dynamics_model_atari_pong
```

Action space collapses consistently. This can be mitigated by ensuring that any pooling of our action encoder features happens after we take the residual. Otherwise, we're looking at frame level movement.

How can we improve?
- Running ablations on batch size
- Predicting action chunks instead of singletons. We learn both a chunk action and a sequence of frames that follows it.
  - Why? This improves our IID assumption.
- Using latent prediction for our action model. Instead of pixel space movement, can we predict how the semantics of the frames change?
  - Could even just use a JEPA as the base model
- I want to stick to unlabeled data as much as possible. I think labels are helpful but cut off our ability to look at any video and translate it.
- Could programmatically boost the importance of the ball prediction
- LPIPS

CUDA_VISIBLE_DEVICES=5 python -m scripts.dynamics_model.train --dataset_type atari_pong --atari_pong_data_dir data/atari_pong --tokenizer_checkpoint_path fsq_tokenizer_atari_pong/checkpoint_epoch1_batch3000.pt --image_size 128 --patch_size 4 --num_images_in_video 5 --batch_size 4 --gradient_accumulation_steps 8 --num_epochs 10 --save_dir dynamics_model_atari_pong_action_103m_5_frames_l1_residual_ --checkpoint_dir dynamics_model_atari_pong_action_103m_5_frames_l1_residual_ --action_d_model 512 --action_num_transformer_layers 8 --action_num_heads 8 --action_latent_dim 64 --logging-backend tensorboard --tensorboard-dir runs --atari_pong_require_full_gameplay

I am fairly confident that the action model is just learning simple training signals. I want to try predicting purely on the residuals ![loss mask visualized](public/loss_mask_visualization_pong.png).

CUDA_VISIBLE_DEVICES=0 python -m scripts.dynamics_model.train --dataset_type atari_pong --atari_pong_data_dir data/atari_pong --tokenizer_checkpoint_path fsq_tokenizer_atari_pong/checkpoint_epoch1_batch3000.pt --image_size 128 --patch_size 4 --num_images_in_video 5 --batch_size 4 --gradient_accumulation_steps 8 --num_epochs 1000 --dataset_limit 100 --save_dir dynamics_model_atari_pong_action_103m_5_frames_naive_direct --checkpoint_dir dynamics_model_atari_pong_action_103m_5_frames_naive_direct --action_d_model 512 --action_num_transformer_layers 8 --action_num_heads 8 --action_latent_dim 64 --logging-backend tensorboard --tensorboard-dir runs --atari_pong_require_full_gameplay --no-predict-action-residuals --action-decoder-loss l2 --dynamics-token-loss ce --experiment-name naive_direct > naive.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -m scripts.dynamics_model.train --dataset_type atari_pong --atari_pong_data_dir data/atari_pong --tokenizer_checkpoint_path fsq_tokenizer_atari_pong/checkpoint_epoch1_batch3000.pt --image_size 128 --patch_size 4 --num_images_in_video 5 --batch_size 4 --gradient_accumulation_steps 8 --num_epochs 1000 --dataset_limit 100 --save_dir dynamics_model_atari_pong_action_103m_5_frames_naive_residual --checkpoint_dir dynamics_model_atari_pong_action_103m_5_frames_naive_residual --action_d_model 512 --action_num_transformer_layers 8 --action_num_heads 8 --action_latent_dim 64 --logging-backend tensorboard --tensorboard-dir runs --atari_pong_require_full_gameplay --predict-action-residuals --action-decoder-loss l2 --dynamics-token-loss ce --experiment-name naive_residual > naive_residual.log 2>&1

CUDA_VISIBLE_DEVICES=4 python -m scripts.dynamics_model.train --dataset_type atari_pong --atari_pong_data_dir data/atari_pong --tokenizer_checkpoint_path fsq_tokenizer_atari_pong/checkpoint_epoch1_batch3000.pt --image_size 128 --patch_size 4 --num_images_in_video 5 --batch_size 4 --gradient_accumulation_steps 8 --num_epochs 1000 --dataset_limit 100 --save_dir dynamics_model_atari_pong_action_103m_5_frames_clipped_direct --checkpoint_dir dynamics_model_atari_pong_action_103m_5_frames_clipped_direct --action_d_model 512 --action_num_transformer_layers 8 --action_num_heads 8 --action_latent_dim 64 --logging-backend tensorboard --tensorboard-dir runs --atari_pong_require_full_gameplay --no-predict-action-residuals --action-decoder-loss clipped_l2 --action-l2-clip-c 10 --dynamics-token-loss clipped_ce --dynamics-ce-clip-c 0.03 --experiment-name clipped_direct_zero > naive_clipped.log 2>&1

CUDA_VISIBLE_DEVICES=5 python -m scripts.dynamics_model.train --dataset_type atari_pong --atari_pong_data_dir data/atari_pong --tokenizer_checkpoint_path fsq_tokenizer_atari_pong/checkpoint_epoch1_batch3000.pt --image_size 128 --patch_size 4 --num_images_in_video 5 --batch_size 4 --gradient_accumulation_steps 8 --num_epochs 1000 --dataset_limit 100 --save_dir dynamics_model_atari_pong_action_103m_5_frames_clipped_residual --checkpoint_dir dynamics_model_atari_pong_action_103m_5_frames_clipped_residual --action_d_model 512 --action_num_transformer_layers 8 --action_num_heads 8 --action_latent_dim 64 --logging-backend tensorboard --tensorboard-dir runs --atari_pong_require_full_gameplay --predict-action-residuals --action-decoder-loss clipped_l2 --action-l2-clip-c 10 --dynamics-token-loss clipped_ce --dynamics-ce-clip-c 0.03 --experiment-name clipped_residual_zero > naive_residual_clipped_zero.log 2>&1

CUDA_VISIBLE_DEVICES=7 python -m scripts.dynamics_model.train --dataset_type atari_pong --atari_pong_data_dir data/atari_pong --tokenizer_checkpoint_path fsq_tokenizer_atari_pong_large_ds/checkpoint_epoch2_batch31250.pt --image_size 84 --patch_size 4 --num_images_in_video 5 --batch_size 8 --gradient_accumulation_steps 8 --num_epochs 1000 --dataset_limit 100 --save_dir dynamics_model_atari_pong_action_103m_5_frames_clipped_residual_new_non_zero --checkpoint_dir dynamics_model_atari_pong_action_103m_5_frames_clipped_residual_new_non_zero --action_d_model 512 --action_num_transformer_layers 8 --action_num_heads 8 --action_latent_dim 64 --logging-backend tensorboard --tensorboard-dir runs --atari_pong_require_full_gameplay --predict-action-residuals --action-decoder-loss clipped_l2 --action-l2-clip-c 10 --dynamics-token-loss clipped_ce --dynamics-ce-clip-c 0.03 --experiment-name clipped_residual_new_head  > naive_residual_clipped_new_head.log 2>&1


CUDA_VISIBLE_DEVICES=6 python -m scripts.dynamics_model.train --dataset_type atari_pong --atari_pong_data_dir data/atari_pong --tokenizer_checkpoint_path fsq_tokenizer_atari_pong_large_ds/checkpoint_epoch2_batch31250.pt --image_size 84 --patch_size 4 --num_images_in_video 5 --batch_size 8 --gradient_accumulation_steps 2 --num_epochs 1000 --dataset_limit 1_000_000 --save_dir dynamics_model_pong_w_tokenizer_v2_lower_lr --checkpoint_dir dynamics_model_pong_w_tokenizer_v2_lower_lr --action_d_model 256 --action_num_transformer_layers 8 --action_num_heads 8 --action_latent_dim 32 --logging-backend tensorboard --tensorboard-dir runs --atari_pong_require_full_gameplay --experiment-name dynamics_model_pong_w_tokenizer_v2_lower_lr  > dynamics_model_pong_w_tokenizer_v2_lower_lr.log 2>&1

CUDA_VISIBLE_DEVICES=4 python -m scripts.dynamics_model.train --dataset_type atari_pong --atari_pong_data_dir data/atari_pong --tokenizer_checkpoint_path fsq_tokenizer_atari_pong_large_ds/checkpoint_epoch2_batch31250.pt --image_size 84 --patch_size 4 --num_images_in_video 16 --batch_size 4 --gradient_accumulation_steps 6 --num_epochs 2 --dataset_limit 1_000_000 --save_dir dynamics_model_pong_w_tokenizer_v2_256_action_w_corrected_upsample_dim_and_patch_16 --checkpoint_dir dynamics_model_pong_w_tokenizer_v2_256_action_w_corrected_upsample_dim_and_patch_16 --action_d_model 512 --action_num_transformer_layers 8 --action_num_heads 8 --action_latent_dim 32 --logging-backend tensorboard --tensorboard-dir runs --atari_pong_require_full_gameplay --experiment-name dynamics_model_pong_w_tokenizer_v2_256_action_w_corrected_upsample_dim_and_patch_16  > dynamics_model_pong_w_tokenizer_v2_256_action_w_corrected_upsample_dim_and_patch_16.log 2>&1


Retrain decoder


- Early stopping for LAM
- Norms and residuals fixed in LAM
- Remove pre-norm on PatchEmbedding
- Switch to RMSNorm

dynamics_model_pong_w_tokenizer_v2 has some crazy results from pre-action model updates. Might be worth reverting action _model to 04-28-upgrade_video_tokenizer_with_swiglu_rmsnorm_nonlinear_upsampler. TBD, need to train the others to same extent.

CUDA_VISIBLE_DEVICES=5 python -m scripts.dynamics_model.train --dataset_type atari_pong --atari_pong_data_dir data/atari_pong --tokenizer_checkpoint_path fsq_tokenizer_atari_pong_large_ds/checkpoint_epoch2_batch31250.pt --image_size 84 --patch_size 4 --num_images_in_video 5 --batch_size 8 --gradient_accumulation_steps 6 --num_epochs 2 --dataset_limit 1_000_000 --save_dir dynamics_model_pong_w_tokenizer_v2_256_scheduled --checkpoint_dir dynamics_model_pong_w_tokenizer_v2_256_scheduled --action_d_model 512 --action_num_transformer_layers 8 --action_num_heads 8 --action_latent_dim 32 --logging-backend tensorboard --tensorboard-dir runs --atari_pong_require_full_gameplay --experiment-name dynamics_model_pong_w_tokenizer_v2_256_scheduled >> dynamics_model_pong_w_tokenizer_v2_256_scheduled.log 2>&1