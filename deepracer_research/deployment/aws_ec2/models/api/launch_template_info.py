from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LaunchTemplateInfo:
    """Information about an EC2 launch template

    Parameters
    ----------
    template_id : str
        Launch template ID.
    template_name : str
        Launch template name.
    version : str, optional
        Template version, by default "$Latest".
    description : str, optional
        Template description, by default None.
    """

    template_id: str
    template_name: str
    version: str = "$Latest"
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation.
        """
        return {"LaunchTemplateId": self.template_id, "LaunchTemplateName": self.template_name, "Version": self.version}
