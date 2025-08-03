from typing import Any, Dict, Optional

from botocore.exceptions import ClientError

from deepracer_research.deployment.aws_ec2.config.aws_config import AWSConfig
from deepracer_research.deployment.aws_ec2.models.api import EC2ApiError
from deepracer_research.utils.logger import error, info, warning


class AMIFinder:
    """Utility for finding appropriate AMIs for EC2 instances"""

    def __init__(self, aws_config: AWSConfig):
        """Initialize AMI finder.

        Parameters
        ----------
        aws_config : AWSConfig
            AWS configuration.
        """
        self.aws_config = aws_config

        import boto3

        session_kwargs = aws_config.get_boto3_session_kwargs()
        self.session = boto3.Session(**session_kwargs)
        self.ec2_client = self.session.client("ec2")

    def find_ubuntu_ami(
        self, version: str = "22.04", architecture: str = "x86_64", virtualization_type: str = "hvm"
    ) -> Optional[str]:
        """Find the latest Ubuntu AMI.

        Parameters
        ----------
        version : str, optional
            Ubuntu version, by default "22.04".
        architecture : str, optional
            Architecture (x86_64 or arm64), by default "x86_64".
        virtualization_type : str, optional
            Virtualization type, by default "hvm".

        Returns
        -------
        str, optional
            AMI ID if found, None otherwise.

        Raises
        ------
        EC2ApiError
            If AMI search fails.
        """
        try:
            name_pattern = f"ubuntu/images/{virtualization_type}-ssd/ubuntu-*-{version}-*-server-*"

            filters = [
                {"Name": "name", "Values": [name_pattern]},
                {"Name": "state", "Values": ["available"]},
                {"Name": "architecture", "Values": [architecture]},
                {"Name": "virtualization-type", "Values": [virtualization_type]},
                {"Name": "root-device-type", "Values": ["ebs"]},
            ]

            response = self.ec2_client.describe_images(Filters=filters, Owners=["099720109477"])

            if not response["Images"]:
                warning(
                    "No Ubuntu AMI found",
                    extra={"version": version, "architecture": architecture, "region": self.aws_config.region.value},
                )
                return None

            images = sorted(response["Images"], key=lambda x: x["CreationDate"], reverse=True)

            latest_ami = images[0]
            ami_id = latest_ami["ImageId"]

            info(
                "Found Ubuntu AMI",
                extra={
                    "ami_id": ami_id,
                    "name": latest_ami["Name"],
                    "creation_date": latest_ami["CreationDate"],
                    "version": version,
                    "architecture": architecture,
                },
            )

            return ami_id

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "ClientError")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            error(
                "Failed to find Ubuntu AMI",
                extra={
                    "error_code": error_code,
                    "error_message": error_message,
                    "version": version,
                    "architecture": architecture,
                },
            )

            raise EC2ApiError(f"Failed to find Ubuntu AMI: {error_message}", error_code=error_code, operation="find_ubuntu_ami")

    def find_amazon_linux_ami(self, version: str = "2", architecture: str = "x86_64") -> Optional[str]:
        """Find the latest Amazon Linux AMI.

        Parameters
        ----------
        version : str, optional
            Amazon Linux version ("1" or "2"), by default "2".
        architecture : str, optional
            Architecture (x86_64 or arm64), by default "x86_64".

        Returns
        -------
        str, optional
            AMI ID if found, None otherwise.

        Raises
        ------
        EC2ApiError
            If AMI search fails.
        """
        try:
            if version == "2":
                name_pattern = "amzn2-ami-hvm-*"
                owner_id = "137112412989"
            else:
                name_pattern = "amzn-ami-hvm-*"
                owner_id = "137112412989"

            filters = [
                {"Name": "name", "Values": [name_pattern]},
                {"Name": "owner-id", "Values": [owner_id]},
                {"Name": "state", "Values": ["available"]},
                {"Name": "architecture", "Values": [architecture]},
                {"Name": "virtualization-type", "Values": ["hvm"]},
                {"Name": "root-device-type", "Values": ["ebs"]},
            ]

            response = self.ec2_client.describe_images(Filters=filters, Owners=[owner_id])

            if not response["Images"]:
                warning(
                    "No Amazon Linux AMI found",
                    extra={"version": version, "architecture": architecture, "region": self.aws_config.region.value},
                )
                return None

            images = sorted(response["Images"], key=lambda x: x["CreationDate"], reverse=True)

            latest_ami = images[0]
            ami_id = latest_ami["ImageId"]

            info(
                "Found Amazon Linux AMI",
                extra={
                    "ami_id": ami_id,
                    "name": latest_ami["Name"],
                    "creation_date": latest_ami["CreationDate"],
                    "version": version,
                    "architecture": architecture,
                },
            )

            return ami_id

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "ClientError")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            error(
                "Failed to find Amazon Linux AMI",
                extra={
                    "error_code": error_code,
                    "error_message": error_message,
                    "version": version,
                    "architecture": architecture,
                },
            )

            raise EC2ApiError(
                f"Failed to find Amazon Linux AMI: {error_message}", error_code=error_code, operation="find_amazon_linux_ami"
            )

    def get_ami_details(self, ami_id: str) -> Dict[str, Any]:
        """Get detailed information about an AMI.

        Parameters
        ----------
        ami_id : str
            AMI ID.

        Returns
        -------
        Dict[str, Any]
            AMI details.

        Raises
        ------
        EC2ApiError
            If AMI details cannot be retrieved.
        """
        try:
            response = self.ec2_client.describe_images(ImageIds=[ami_id])

            if not response["Images"]:
                raise EC2ApiError(f"AMI {ami_id} not found", error_code="AMINotFound", operation="get_ami_details")

            ami = response["Images"][0]

            info("Retrieved AMI details", extra={"ami_id": ami_id, "name": ami["Name"], "state": ami["State"]})

            return ami

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "ClientError")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            error(
                "Failed to get AMI details", extra={"ami_id": ami_id, "error_code": error_code, "error_message": error_message}
            )

            raise EC2ApiError(
                f"Failed to get AMI details for {ami_id}: {error_message}", error_code=error_code, operation="get_ami_details"
            )

    def find_deepracer_optimized_ami(self, architecture: str = "x86_64") -> Optional[str]:
        """Find AMI optimized for DeepRacer workloads

        Parameters
        ----------
        architecture : str, optional
            Architecture (x86_64 or arm64), by default "x86_64".

        Returns
        -------
        str, optional
            AMI ID if found, None otherwise.
        """
        info("Finding DeepRacer-optimized AMI", extra={"architecture": architecture})

        return self.find_ubuntu_ami(version="22.04", architecture=architecture)
