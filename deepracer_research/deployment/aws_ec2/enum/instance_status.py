from enum import StrEnum


class EC2InstanceStatus(StrEnum):
    """EC2 instance status enumeration"""

    PENDING = "pending"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting-down"
    TERMINATED = "terminated"
    STOPPING = "stopping"
    STOPPED = "stopped"

    @property
    def is_ready_for_ssh(self) -> bool:
        """Check if instance is ready for SSH connections.

        Returns
        -------
        bool
            True if instance can accept SSH connections.
        """
        return self == self.RUNNING

    @property
    def is_billable(self) -> bool:
        """Check if instance is consuming billable resources.

        Returns
        -------
        bool
            True if instance is consuming compute resources.
        """
        return self in [self.PENDING, self.RUNNING]

    @property
    def is_transitioning(self) -> bool:
        """Check if instance is in a transitional state.

        Returns
        -------
        bool
            True if instance is changing states.
        """
        return self in [self.PENDING, self.SHUTTING_DOWN, self.STOPPING]

    @property
    def is_terminal(self) -> bool:
        """Check if instance is in a terminal state.

        Returns
        -------
        bool
            True if instance cannot be restarted.
        """
        return self == self.TERMINATED

    @property
    def can_be_started(self) -> bool:
        """Check if instance can be started.

        Returns
        -------
        bool
            True if instance can be started from current state.
        """
        return self == self.STOPPED

    @property
    def can_be_stopped(self) -> bool:
        """Check if instance can be stopped.

        Returns
        -------
        bool
            True if instance can be stopped from current state.
        """
        return self == self.RUNNING

    @property
    def can_be_terminated(self) -> bool:
        """Check if instance can be terminated.

        Returns
        -------
        bool
            True if instance can be terminated from current state.
        """
        return self not in [self.TERMINATED, self.SHUTTING_DOWN]
