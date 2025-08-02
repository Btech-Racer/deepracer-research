from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from deepracer_research.deployment.aws_ec2.enum.deployment_mode import EC2DeploymentMode
from deepracer_research.deployment.aws_ec2.enum.instance_type import EC2InstanceType
from deepracer_research.deployment.aws_ec2.enum.region import AWSRegion


@dataclass
class EC2InstanceConfig:
    """Configuration for EC2 instance creation

    Parameters
    ----------
    instance_type : EC2InstanceType
        EC2 instance type to launch.
    region : AWSRegion
        AWS region for instance deployment.
    ami_id : str, optional
        Amazon Machine Image ID. If not provided, will use latest Ubuntu LTS, by default None.
    key_name : str, optional
        EC2 key pair name for SSH access, by default None.
    deployment_mode : EC2DeploymentMode, optional
        Deployment mode affecting pricing, by default ON_DEMAND.
    vpc_id : str, optional
        VPC ID for instance placement, by default None (uses default VPC).
    subnet_id : str, optional
        Subnet ID for instance placement, by default None (uses default subnet).
    security_group_ids : List[str], optional
        List of security group IDs, by default empty list.
    instance_name : str, optional
        Instance name tag, by default None.
    user_data_script : str, optional
        User data script for instance initialization, by default None.
    install_deepracer_cloud : bool, optional
        Whether to automatically install DeepRacer-for-Cloud, by default True.
    s3_bucket_name : str, optional
        S3 bucket name for model storage, by default None.
    iam_instance_profile : str, optional
        IAM instance profile name, by default None.
    ebs_volume_size : int, optional
        Root EBS volume size in GB, by default 100.
    ebs_volume_type : str, optional
        EBS volume type, by default "gp3".
    enable_detailed_monitoring : bool, optional
        Enable detailed CloudWatch monitoring, by default False.
    spot_max_price : float, optional
        Maximum spot price (only for SPOT deployment mode), by default None.
    tags : Dict[str, str], optional
        Additional tags for the instance, by default empty dict.
    environment_variables : Dict[str, str], optional
        Environment variables to set on the instance, by default empty dict.
    """

    instance_type: EC2InstanceType
    region: AWSRegion
    ami_id: Optional[str] = None
    key_name: Optional[str] = None
    deployment_mode: EC2DeploymentMode = EC2DeploymentMode.ON_DEMAND

    vpc_id: Optional[str] = None
    subnet_id: Optional[str] = None
    security_group_ids: List[str] = field(default_factory=list)

    instance_name: Optional[str] = None
    user_data_script: Optional[str] = None

    install_deepracer_cloud: bool = True
    s3_bucket_name: Optional[str] = None

    iam_instance_profile: Optional[str] = None

    ebs_volume_size: int = 100
    ebs_volume_type: str = "gp3"

    enable_detailed_monitoring: bool = False

    spot_max_price: Optional[float] = None

    tags: Dict[str, str] = field(default_factory=dict)
    environment_variables: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for AWS API calls.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation suitable for AWS EC2 API.
        """
        config = {"ImageId": self.ami_id, "InstanceType": self.instance_type.value, "MinCount": 1, "MaxCount": 1}

        if self.key_name:
            config["KeyName"] = self.key_name

        if self.security_group_ids:
            config["SecurityGroupIds"] = self.security_group_ids

        if self.subnet_id:
            config["SubnetId"] = self.subnet_id

        if self.user_data_script:
            config["UserData"] = self.user_data_script

        if self.iam_instance_profile:
            config["IamInstanceProfile"] = {"Name": self.iam_instance_profile}

        config["BlockDeviceMappings"] = [
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {"VolumeSize": self.ebs_volume_size, "VolumeType": self.ebs_volume_type, "DeleteOnTermination": True},
            }
        ]

        config["Monitoring"] = {"Enabled": self.enable_detailed_monitoring}

        tag_specifications = []
        instance_tags = self.tags.copy()
        if self.instance_name:
            instance_tags["Name"] = self.instance_name

        if instance_tags:
            tag_specifications.append(
                {"ResourceType": "instance", "Tags": [{"Key": k, "Value": v} for k, v in instance_tags.items()]}
            )

        if tag_specifications:
            config["TagSpecifications"] = tag_specifications

        return config

    def get_spot_specification(self) -> Dict[str, Any]:
        """Get spot instance specification.

        Returns
        -------
        Dict[str, Any]
            Spot instance request specification.

        Raises
        ------
        ValueError
            If deployment mode is not SPOT.
        """
        if self.deployment_mode != EC2DeploymentMode.SPOT:
            raise ValueError("Spot specification only available for SPOT deployment mode")

        spec = {"LaunchSpecification": self.to_dict(), "Type": "one-time", "InstanceCount": 1}

        if self.spot_max_price:
            spec["SpotPrice"] = str(self.spot_max_price)

        return spec

    def validate(self) -> None:
        """Validate the configuration parameters.

        Raises
        ------
        ValueError
            If any configuration parameter is invalid.
        """
        if self.ebs_volume_size < 8:
            raise ValueError("EBS volume size must be at least 8GB")

        if self.install_deepracer_cloud and self.ebs_volume_size < 40:
            raise ValueError("DeepRacer installation requires at least 40GB EBS volume")

        if not self.instance_type.is_suitable_for_training and self.install_deepracer_cloud:
            raise ValueError(f"Instance type {self.instance_type.value} may not be suitable for DeepRacer training")

        if self.deployment_mode == EC2DeploymentMode.SPOT:
            if not self.instance_type.is_suitable_for_training:
                pass

        if self.vpc_id and not self.vpc_id.startswith("vpc-"):
            raise ValueError("VPC ID must start with 'vpc-'")

        if self.subnet_id and not self.subnet_id.startswith("subnet-"):
            raise ValueError("Subnet ID must start with 'subnet-'")

        for sg_id in self.security_group_ids:
            if not sg_id.startswith("sg-"):
                raise ValueError(f"Security group ID must start with 'sg-': {sg_id}")

        if self.ami_id and not self.ami_id.startswith("ami-"):
            raise ValueError("AMI ID must start with 'ami-'")

    @classmethod
    def for_deepracer_training(
        cls,
        region: AWSRegion = AWSRegion.US_EAST_1,
        instance_type: EC2InstanceType = EC2InstanceType.G4DN_XLARGE,
        s3_bucket_name: Optional[str] = None,
        key_name: Optional[str] = None,
        instance_name: str = "deepracer-training",
    ) -> "EC2InstanceConfig":
        """Create configuration optimized for DeepRacer training.

        Parameters
        ----------
        region : AWSRegion, optional
            AWS region, by default US_EAST_1.
        instance_type : EC2InstanceType, optional
            Instance type, by default G4DN_XLARGE.
        s3_bucket_name : str, optional
            S3 bucket name, by default None.
        key_name : str, optional
            EC2 key pair name, by default None.
        instance_name : str, optional
            Instance name, by default "deepracer-training".

        Returns
        -------
        EC2InstanceConfig
            Configuration optimized for training.
        """
        return cls(
            instance_type=instance_type,
            region=region,
            deployment_mode=EC2DeploymentMode.ON_DEMAND,
            instance_name=instance_name,
            s3_bucket_name=s3_bucket_name,
            key_name=key_name,
            install_deepracer_cloud=True,
            ebs_volume_size=100,
            enable_detailed_monitoring=True,
            tags={"Purpose": "DeepRacer Training", "Project": "DeepRacer Research"},
        )

    @classmethod
    def for_deepracer_evaluation(
        cls,
        region: AWSRegion = AWSRegion.US_EAST_1,
        instance_type: EC2InstanceType = EC2InstanceType.C5_XLARGE,
        key_name: Optional[str] = None,
        instance_name: str = "deepracer-evaluation",
    ) -> "EC2InstanceConfig":
        """Create configuration optimized for DeepRacer evaluation.

        Parameters
        ----------
        region : AWSRegion, optional
            AWS region, by default US_EAST_1.
        instance_type : EC2InstanceType, optional
            Instance type, by default C5_XLARGE.
        key_name : str, optional
            EC2 key pair name, by default None.
        instance_name : str, optional
            Instance name, by default "deepracer-evaluation".

        Returns
        -------
        EC2InstanceConfig
            Configuration optimized for evaluation.
        """
        return cls(
            instance_type=instance_type,
            region=region,
            deployment_mode=EC2DeploymentMode.ON_DEMAND,
            instance_name=instance_name,
            key_name=key_name,
            install_deepracer_cloud=True,
            ebs_volume_size=50,
            enable_detailed_monitoring=False,
            tags={"Purpose": "DeepRacer Evaluation", "Project": "DeepRacer Research"},
        )

    @classmethod
    def for_development(
        cls,
        region: AWSRegion = AWSRegion.US_EAST_1,
        instance_type: EC2InstanceType = EC2InstanceType.G4DN_XLARGE,
        key_name: Optional[str] = None,
        instance_name: str = "deepracer-dev",
    ) -> "EC2InstanceConfig":
        """Create configuration optimized for development/experimentation.

        Parameters
        ----------
        region : AWSRegion, optional
            AWS region, by default US_EAST_1.
        instance_type : EC2InstanceType, optional
            Instance type, by default G4DN_XLARGE.
        key_name : str, optional
            EC2 key pair name, by default None.
        instance_name : str, optional
            Instance name, by default "deepracer-dev".

        Returns
        -------
        EC2InstanceConfig
            Configuration optimized for development.
        """
        return cls(
            instance_type=instance_type,
            region=region,
            deployment_mode=EC2DeploymentMode.SPOT,
            instance_name=instance_name,
            key_name=key_name,
            install_deepracer_cloud=True,
            ebs_volume_size=80,
            enable_detailed_monitoring=False,
            tags={"Purpose": "DeepRacer Development", "Project": "DeepRacer Research", "Environment": "Development"},
        )
