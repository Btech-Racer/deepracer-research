from typing import Optional


class AWSError(Exception):
    """Base exception for AWS-related errors"""

    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class EC2ApiError(AWSError):
    """Exception for EC2 API-specific errors"""

    def __init__(
        self, message: str, error_code: Optional[str] = None, instance_id: Optional[str] = None, operation: Optional[str] = None
    ):
        super().__init__(message, error_code)
        self.instance_id = instance_id
        self.operation = operation
