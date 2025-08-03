import logging
from typing import Optional

import boto3
from botocore.config import Config


def get_deepracer_config(region_name: str = "us-east-1") -> Config:
    """Get optimized boto3 config for DeepRacer service.

    Parameters
    ----------
    region_name : str, optional
        AWS region name, by default "us-east-1"

    Returns
    -------
    Config
        Boto3 configuration optimized for DeepRacer
    """
    return Config(
        region_name=region_name,
        retries={"max_attempts": 3, "mode": "adaptive"},
        signature_version="v4",
        parameter_validation=True,
        s3={"addressing_style": "virtual"},
    )


def configure_aws_logging(level: int = logging.WARNING) -> None:
    """Configure AWS SDK logging to suppress verbose warnings.

    Parameters
    ----------
    level : int, optional
        Logging level for AWS SDK components, by default logging.WARNING
    """
    logging.getLogger("botocore.loaders").setLevel(level)
    logging.getLogger("botocore.utils").setLevel(level)
    logging.getLogger("botocore.endpoint").setLevel(level)
    logging.getLogger("boto3.resources").setLevel(level)


def get_aws_session(region_name: Optional[str] = None, profile_name: Optional[str] = None) -> boto3.Session:
    """Get a configured AWS session with proper logging.

    Parameters
    ----------
    region_name : Optional[str], optional
        AWS region name, by default None
    profile_name : Optional[str], optional
        AWS profile name, by default None

    Returns
    -------
    boto3.Session
        Configured AWS session
    """
    configure_aws_logging()

    session_kwargs = {}
    if region_name:
        session_kwargs["region_name"] = region_name
    if profile_name:
        session_kwargs["profile_name"] = profile_name

    return boto3.Session(**session_kwargs)


def get_deepracer_client(
    region_name: str = "us-east-1", profile_name: Optional[str] = None, aws_session: Optional[boto3.Session] = None
) -> boto3.client:
    """Get a properly configured DeepRacer client.

    Parameters
    ----------
    region_name : str, optional
        AWS region name, by default "us-east-1"
    profile_name : Optional[str], optional
        AWS profile name, by default None
    aws_session : Optional[boto3.Session], optional
        Existing AWS session, by default None

    Returns
    -------
    boto3.client
        Properly configured DeepRacer client
    """
    config = get_deepracer_config(region_name)

    if aws_session:
        return aws_session.client("deepracer", config=config)
    else:
        session = get_aws_session(region_name, profile_name)
        return session.client("deepracer", config=config)


def get_s3_client(
    region_name: str = "us-east-1", profile_name: Optional[str] = None, aws_session: Optional[boto3.Session] = None
) -> boto3.client:
    """Get a properly configured S3 client.

    Parameters
    ----------
    region_name : str, optional
        AWS region name, by default "us-east-1"
    profile_name : Optional[str], optional
        AWS profile name, by default None
    aws_session : Optional[boto3.Session], optional
        Existing AWS session, by default None

    Returns
    -------
    boto3.client
        Properly configured S3 client
    """
    config = get_deepracer_config(region_name)

    if aws_session:
        return aws_session.client("s3", config=config)
    else:
        session = get_aws_session(region_name, profile_name)
        return session.client("s3", config=config)


configure_aws_logging()
