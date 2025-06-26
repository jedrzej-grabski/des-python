import numpy as np
from numpy.typing import NDArray
from collections import deque
from typing import List, Union


class RingBuffer:
    """
    A fixed-size buffer that overwrites oldest data when full.
    Implements ring buffer functionality using collections.deque.
    """

    def __init__(self, size: int) -> None:
        """
        Initialize a ring buffer with a fixed size.

        Args:
            size: Maximum number of elements in the buffer
        """
        self.size = size
        self.buffer = deque(maxlen=size)
        self.is_full = False

    def push(self, item: float | NDArray[np.float64]) -> None:
        """
        Add an item to the buffer.

        Args:
            item: The value to add to the bufferW
        """
        if len(self.buffer) == self.size:
            self.is_full = True

        self.buffer.append(item)

    def push_all(self, items: list[float | NDArray[np.float64]]) -> None:
        """
        Add multiple items to the buffer.

        Args:
            items: The values to add to the buffer
        """
        for item in items:
            self.push(item)

    def get(self, idx: int) -> float:
        """
        Get an item from the buffer by its current position.

        Args:
            idx: The index of the item to retrieve

        Returns:
            The value at the specified index
        """
        if idx < 0 or idx >= self.capacity():
            raise IndexError("Buffer index out of range")

        return self.buffer[idx]

    def peek(self) -> NDArray[np.float64]:
        """
        Get all items currently in the buffer.

        Returns:
            Array containing all buffer values in order
        """
        return np.array(self.buffer)

    def capacity(self) -> int:
        """
        Get the current number of items in the buffer.

        Returns:
            The number of items currently in the buffer
        """
        return len(self.buffer)

    def clear(self) -> None:
        """
        Clear the buffer.
        """
        self.buffer.clear()
        self.is_full = False

    def is_empty(self) -> bool:
        """
        Check if the buffer is empty.

        Returns:
            True if the buffer is empty, False otherwise
        """
        return len(self.buffer) == 0
