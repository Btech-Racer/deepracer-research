from dataclasses import dataclass
from typing import Optional


@dataclass
class EC2KeyPairInfo:
    """EC2 key pair information

    Parameters
    ----------
    key_name : str
        Key pair name.
    key_fingerprint : str, optional
        Key fingerprint, by default None.
    key_material : str, optional
        Private key material (only for newly created keys), by default None.
    """

    key_name: str
    key_fingerprint: Optional[str] = None
    key_material: Optional[str] = None

    def validate(self) -> None:
        """Validate key pair information.

        Raises
        ------
        ValueError
            If key name is missing.
        """
        if not self.key_name:
            raise ValueError("Key name is required")
