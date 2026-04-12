
class Progress:
    """
    Progress class manages status updates and notifies registered callbacks.
    """
    def __init__(self):
        """Initialize the Progress object with an empty callback list."""
        self.callbacks = []

    def add_callback(self, cb):
        """Add a callback function to be notified on status updates."""
        self.callbacks.append(cb)

    def emit(self, status):
        """
        Emit a status update to all registered callbacks.
        Args:
            status (dict): Status information to send.
        """
        for cb in self.callbacks:
            cb(status)
