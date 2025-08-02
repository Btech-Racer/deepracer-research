import json
import uuid
from datetime import datetime

import boto3
from botocore.exceptions import ClientError

from deepracer_research.deployment.aws_ec2.enum.region import AWSRegion
from deepracer_research.utils import error


def create_deepracer_s3_bucket(
    project_name: str, model_name: str, region: str = AWSRegion.US_EAST_1.value, profile: str = "default"
) -> dict[str, str]:
    """Create S3 bucket for DeepRacer with terminal output.

    Parameters
    ----------
    project_name : str
        Project name for bucket naming
    model_name : str
        Model name for bucket naming
    region : str, optional
        AWS region, by default "us-east-1"
    profile : str, optional
        AWS profile, by default "default"

    Returns
    -------
    dict[str, str]
        Dictionary with bucket name and URI
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_id = str(uuid.uuid4())[:8]

    bucket_name = f"deepracer-{project_name}-{model_name}-{timestamp}-{unique_id}".lower()
    bucket_name = bucket_name.replace("_", "-")

    try:
        session = boto3.Session(profile_name=profile)
        s3_client = session.client("s3", region_name=region)

        # Create the bucket
        if region == "us-east-1":
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": region})

        deepracer_policy = {
            "Version": "2012-10-17",
            "Id": "AwsDeepracerServiceAccess",
            "Statement": [
                {
                    "Sid": "Stmt1753912486517",
                    "Effect": "Allow",
                    "Principal": {"Service": "deepracer.amazonaws.com"},
                    "Action": [
                        "s3:GetObjectAcl",
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:PutObjectAcl",
                        "s3:ListBucket",
                        "s3:ListBucketVersions",
                    ],
                    "Resource": [f"arn:aws:s3:::{bucket_name}", f"arn:aws:s3:::{bucket_name}/*"],
                }
            ],
        }

        # Apply the bucket policy
        s3_client.put_bucket_policy(Bucket=bucket_name, Policy=json.dumps(deepracer_policy))

        bucket_info = {"name": bucket_name, "uri": f"s3://{bucket_name}"}

        return bucket_info
    except ClientError as e:
        error_msg = f"Failed to create S3 bucket: {e}"
        error(f"{error_msg}")
        raise Exception(error_msg)
