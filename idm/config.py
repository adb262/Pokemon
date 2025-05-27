from truss_train import definitions
from truss.base import truss_config

BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda11.8-cudnn9-devel"

training_runtime = definitions.Runtime(
    start_commands=[
        "bash ./run.sh",
    ],
    environment_variables={
        "AWS_REGION": definitions.SecretReference(name="aws_region"),
        "S3_BUCKET_NAME": definitions.SecretReference(name="aws_data_bucket_name"),
        "AWS_ACCESS_KEY_ID": definitions.SecretReference(name="aws_access_key_id"),
        "AWS_SECRET_ACCESS_KEY": definitions.SecretReference(name="aws_secret_access_key"),
        "WANDB_API_KEY": definitions.SecretReference(name="wandb_api_key"),
    },
    enable_cache=True,
)

training_compute = definitions.Compute(
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.A10G,
        count=1,
    ),
)

my_training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime
)

first_project_with_job = definitions.TrainingProject(
    name="vqvae-test",
    job=my_training_job
)
