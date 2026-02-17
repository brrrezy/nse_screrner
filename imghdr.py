"""
Minimal shim for Python 3.13 environments where the stdlib `imghdr`
module has been removed.

Streamlit still imports `imghdr` internally to guess image types.
For our app we don't rely on that behaviour, so a very small stub
implementation is enough to satisfy the import.
"""

from typing import Optional, Union, BinaryIO


def what(file: Union[str, BinaryIO], h: Optional[bytes] = None) -> Optional[str]:
    """
    Approximate signature of stdlib imghdr.what().
    We simply return None, which means "unknown image type".

    This is sufficient for Streamlit to run; it will fall back to
    other mechanisms when displaying images, and our app does not
    rely on automatic type detection.
    """

    return None

