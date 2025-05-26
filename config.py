from truss_train import definitions
from truss.base import truss_config

BASE_IMAGE = "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04"

training_runtime = definitions.Runtime(
    start_commands=[  # Example: list of commands to run your training script
        # Install pip and uv
        "/bin/sh -c './run.sh'",
    ],
    environment_variables={
        # Secrets (ensure these are configured in your Baseten workspace)
        "AWS_REGION": definitions.SecretReference(name="aws_region"),
        "S3_BUCKET_NAME": definitions.SecretReference(name="aws_data_bucket_name"),
        "AWS_ACCESS_KEY_ID": definitions.SecretReference(name="aws_access_key_id"),
        "AWS_SECRET_ACCESS_KEY": definitions.SecretReference(name="aws_secret_access_key"),
    },
    enable_cache=True,
)

# 3. Define the Compute Resources for the Training Job
training_compute = definitions.Compute(
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.H100,
        count=1,
    ),
)

# 4. Define the Training Job
# This brings together the image, compute, and runtime configurations.
my_training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime
)


# This config will be pushed using the Truss CLI.
# The association of the job to the project happens at the time of push.
first_project_with_job = definitions.TrainingProject(
    name="vqvae-test",
    job=my_training_job
)
