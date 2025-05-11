import numpy as np
from numpy.typing import NDArray


class RingBuffer:
    """
    A fixed-size buffer that overwrites oldest data when full.
    Emulates the ringbuffer functionality from R.
    """

    def __init__(self, size: int) -> None:
        """
        Initialize a ring buffer with a fixed size.

        Args:
            size: Maximum number of elements in the buffer
        """
        self.size = size
        self.data = np.zeros(size)
        self.index = 0
        self.is_full = False

    def push(self, item: float | NDArray[np.float64]) -> None:
        """
        Add an item to the buffer.

        Args:
            item: The value to add to the buffer
        """
        self.data[self.index] = item
        self.index = (self.index + 1) % self.size
        if self.index == 0:
            self.is_full = True

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

        if self.is_full:
            return self.data[(self.index + idx) % self.size]
        else:
            return self.data[idx]

    def peek(self) -> NDArray[np.float64]:
        """
        Get all items currently in the buffer.

        Returns:
            Array containing all buffer values in order
        """
        if self.is_full:
            return np.concatenate([self.data[self.index :], self.data[: self.index]])
        else:
            return self.data[: self.index]

    def capacity(self) -> int:
        """
        Get the current number of items in the buffer.

        Returns:
            The number of items currently in the buffer
        """
        return self.size if self.is_full else self.index

    def clear(self) -> None:
        """
        Clear the buffer.
        """
        self.data = np.zeros(self.size)
        self.index = 0
        self.is_full = False

    def is_empty(self) -> bool:
        """
        Check if the buffer is empty.

        Returns:
            True if the buffer is empty, False otherwise
        """
        return not self.is_full and self.index == 0
