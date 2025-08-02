from enum import StrEnum
from typing import Any, Dict, List


class AWSRegion(StrEnum):
    """Available AWS regions for EC2 deployments"""

    US_EAST_1 = "us-east-1"
    US_EAST_2 = "us-east-2"
    US_WEST_1 = "us-west-1"
    US_WEST_2 = "us-west-2"
    CA_CENTRAL_1 = "ca-central-1"

    EU_WEST_1 = "eu-west-1"
    EU_WEST_2 = "eu-west-2"
    EU_WEST_3 = "eu-west-3"
    EU_CENTRAL_1 = "eu-central-1"
    EU_NORTH_1 = "eu-north-1"

    AP_SOUTHEAST_1 = "ap-southeast-1"
    AP_SOUTHEAST_2 = "ap-southeast-2"
    AP_NORTHEAST_1 = "ap-northeast-1"
    AP_NORTHEAST_2 = "ap-northeast-2"
    AP_SOUTH_1 = "ap-south-1"

    @property
    def display_name(self) -> str:
        """Get the display name for the region.

        Returns
        -------
        str
            Ddisplay name for the region.
        """
        _display_names = {
            self.US_EAST_1: "US East (N. Virginia)",
            self.US_EAST_2: "US East (Ohio)",
            self.US_WEST_1: "US West (N. California)",
            self.US_WEST_2: "US West (Oregon)",
            self.CA_CENTRAL_1: "Canada (Central)",
            self.EU_WEST_1: "Europe (Ireland)",
            self.EU_WEST_2: "Europe (London)",
            self.EU_WEST_3: "Europe (Paris)",
            self.EU_CENTRAL_1: "Europe (Frankfurt)",
            self.EU_NORTH_1: "Europe (Stockholm)",
            self.AP_SOUTHEAST_1: "Asia Pacific (Singapore)",
            self.AP_SOUTHEAST_2: "Asia Pacific (Sydney)",
            self.AP_NORTHEAST_1: "Asia Pacific (Tokyo)",
            self.AP_NORTHEAST_2: "Asia Pacific (Seoul)",
            self.AP_SOUTH_1: "Asia Pacific (Mumbai)",
        }
        return _display_names[self]

    @property
    def timezone(self) -> str:
        """Get the primary timezone for the region.

        Returns
        -------
        str
            Primary timezone identifier.
        """
        _timezones = {
            self.US_EAST_1: "America/New_York",
            self.US_EAST_2: "America/New_York",
            self.US_WEST_1: "America/Los_Angeles",
            self.US_WEST_2: "America/Los_Angeles",
            self.CA_CENTRAL_1: "America/Toronto",
            self.EU_WEST_1: "Europe/Dublin",
            self.EU_WEST_2: "Europe/London",
            self.EU_WEST_3: "Europe/Paris",
            self.EU_CENTRAL_1: "Europe/Berlin",
            self.EU_NORTH_1: "Europe/Stockholm",
            self.AP_SOUTHEAST_1: "Asia/Singapore",
            self.AP_SOUTHEAST_2: "Australia/Sydney",
            self.AP_NORTHEAST_1: "Asia/Tokyo",
            self.AP_NORTHEAST_2: "Asia/Seoul",
            self.AP_SOUTH_1: "Asia/Kolkata",
        }
        return _timezones[self]

    @property
    def supports_gpu_instances(self) -> bool:
        """Check if region supports GPU instances.

        Returns
        -------
        bool
            True if region has GPU instance availability.
        """
        return True

    @property
    def is_government_cloud(self) -> bool:
        """Check if region is AWS GovCloud.

        Returns
        -------
        bool
            True if region is GovCloud.
        """
        return False

    @classmethod
    def get_default(cls) -> "AWSRegion":
        """Get the default AWS region.

        Returns
        -------
        AWSRegion
            Default region (US East 1).
        """
        return cls.US_EAST_1

    @classmethod
    def get_cheapest_regions(cls) -> List["AWSRegion"]:
        """Get regions with typically lower costs.

        Returns
        -------
        List[AWSRegion]
            List of regions with lower costs.
        """
        return [cls.US_EAST_1, cls.US_EAST_2, cls.US_WEST_2]

    @classmethod
    def get_gpu_regions(cls) -> List["AWSRegion"]:
        """Get regions with good GPU instance availability.

        Returns
        -------
        List[AWSRegion]
            List of regions with GPU instances.
        """
        return [region for region in cls if region.supports_gpu_instances]

    @classmethod
    def get_all_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive information about all regions.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping regions to their specifications.
        """
        return {
            region.value: {
                "display_name": region.display_name,
                "timezone": region.timezone,
                "supports_gpu_instances": region.supports_gpu_instances,
                "is_government_cloud": region.is_government_cloud,
            }
            for region in cls
        }
