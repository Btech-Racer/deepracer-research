from typing import Any, Dict, List, Optional

from botocore.exceptions import ClientError

from deepracer_research.deployment.aws_ec2.config.aws_config import AWSConfig
from deepracer_research.deployment.aws_ec2.models.api import EC2ApiError, SecurityGroupRule, VPCInfo
from deepracer_research.utils import error, info, warning


class VPCManager:
    """Manager for VPC, subnet, and security group operations"""

    def __init__(self, aws_config: AWSConfig):
        """Initialize VPC manager.

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
        self.ec2_resource = self.session.resource("ec2")

    def get_default_vpc(self) -> Optional[VPCInfo]:
        """Get the default VPC information.

        Returns
        -------
        VPCInfo, optional
            Default VPC information if available.

        Raises
        ------
        EC2ApiError
            If VPC lookup fails.
        """
        try:
            response = self.ec2_client.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])

            if not response["Vpcs"]:
                info("No default VPC found", extra={"region": self.aws_config.region.value})
                return None

            vpc = response["Vpcs"][0]
            vpc_id = vpc["VpcId"]

            subnet_response = self.ec2_client.describe_subnets(
                Filters=[{"Name": "vpc-id", "Values": [vpc_id]}, {"Name": "default-for-az", "Values": ["true"]}]
            )

            if not subnet_response["Subnets"]:
                warning("No default subnet found in default VPC", extra={"vpc_id": vpc_id})
                return None

            subnet = subnet_response["Subnets"][0]
            subnet_id = subnet["SubnetId"]

            sg_response = self.ec2_client.describe_security_groups(
                Filters=[{"Name": "vpc-id", "Values": [vpc_id]}, {"Name": "group-name", "Values": ["default"]}]
            )

            security_group_ids = []
            if sg_response["SecurityGroups"]:
                security_group_ids = [sg_response["SecurityGroups"][0]["GroupId"]]

            vpc_info = VPCInfo(vpc_id=vpc_id, subnet_id=subnet_id, security_group_ids=security_group_ids)

            info("Found default VPC", extra={"vpc_id": vpc_id, "subnet_id": subnet_id, "security_groups": security_group_ids})

            return vpc_info

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "ClientError")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            error("Failed to get default VPC", extra={"error_code": error_code, "error_message": error_message})

            raise EC2ApiError(f"Failed to get default VPC: {error_message}", error_code=error_code, operation="get_default_vpc")

    def create_security_group(self, name: str, description: str, vpc_id: str, rules: List[SecurityGroupRule]) -> str:
        """Create a security group with specified rules.

        Parameters
        ----------
        name : str
            Security group name.
        description : str
            Security group description.
        vpc_id : str
            VPC ID where to create the security group.
        rules : List[SecurityGroupRule]
            List of security group rules.

        Returns
        -------
        str
            Security group ID.

        Raises
        ------
        EC2ApiError
            If security group creation fails.
        """
        try:
            response = self.ec2_client.create_security_group(GroupName=name, Description=description, VpcId=vpc_id)

            security_group_id = response["GroupId"]

            info("Created security group", extra={"security_group_id": security_group_id, "name": name, "vpc_id": vpc_id})

            if rules:
                self._add_security_group_rules(security_group_id, rules)

            return security_group_id

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "ClientError")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            error(
                "Failed to create security group",
                extra={"name": name, "vpc_id": vpc_id, "error_code": error_code, "error_message": error_message},
            )

            raise EC2ApiError(
                f"Failed to create security group {name}: {error_message}",
                error_code=error_code,
                operation="create_security_group",
            )

    def create_deepracer_security_group(self, vpc_id: str) -> str:
        """Create a security group optimized for DeepRacer workloads.

        Parameters
        ----------
        vpc_id : str
            VPC ID where to create the security group.

        Returns
        -------
        str
            Security group ID.
        """
        name = "deepracer-sg"
        description = "Security group for DeepRacer training instances"

        rules = [
            SecurityGroupRule(protocol="tcp", from_port=22, to_port=22, cidr_blocks=["0.0.0.0/0"], description="SSH access"),
            SecurityGroupRule(
                protocol="tcp", from_port=8888, to_port=8888, cidr_blocks=["0.0.0.0/0"], description="Jupyter notebook access"
            ),
            SecurityGroupRule(
                protocol="tcp", from_port=8100, to_port=8100, cidr_blocks=["0.0.0.0/0"], description="DeepRacer stream viewer"
            ),
            SecurityGroupRule(
                protocol="tcp", from_port=5900, to_port=5901, cidr_blocks=["0.0.0.0/0"], description="VNC access for simulation"
            ),
        ]

        info("Creating DeepRacer-optimized security group", extra={"vpc_id": vpc_id, "rules_count": len(rules)})

        return self.create_security_group(name, description, vpc_id, rules)

    def _add_security_group_rules(self, security_group_id: str, rules: List[SecurityGroupRule]) -> None:
        """Add rules to a security group.

        Parameters
        ----------
        security_group_id : str
            Security group ID.
        rules : List[SecurityGroupRule]
            List of rules to add.
        """
        try:
            ingress_rules = []

            for rule in rules:
                rule_dict = rule.to_dict()
                ingress_rules.append(rule_dict)

            if ingress_rules:
                self.ec2_client.authorize_security_group_ingress(GroupId=security_group_id, IpPermissions=ingress_rules)

                info(
                    "Added security group rules",
                    extra={"security_group_id": security_group_id, "rules_count": len(ingress_rules)},
                )

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "ClientError")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            error(
                "Failed to add security group rules",
                extra={"security_group_id": security_group_id, "error_code": error_code, "error_message": error_message},
            )

            warning("Security group created but rules could not be added", extra={"security_group_id": security_group_id})

    def get_available_subnets(self, vpc_id: str) -> List[Dict[str, Any]]:
        """Get available subnets in a VPC.

        Parameters
        ----------
        vpc_id : str
            VPC ID.

        Returns
        -------
        List[Dict[str, Any]]
            List of subnet information.

        Raises
        ------
        EC2ApiError
            If subnet lookup fails.
        """
        try:
            response = self.ec2_client.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])

            subnets = response["Subnets"]

            info("Found subnets", extra={"vpc_id": vpc_id, "count": len(subnets)})

            return subnets

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "ClientError")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            error("Failed to get subnets", extra={"vpc_id": vpc_id, "error_code": error_code, "error_message": error_message})

            raise EC2ApiError(
                f"Failed to get subnets for VPC {vpc_id}: {error_message}",
                error_code=error_code,
                operation="get_available_subnets",
            )
