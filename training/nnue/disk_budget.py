"""Read-only free-space checks with an optional decimal capacity ceiling."""
from decimal import Decimal
import shutil


def available_bytes(path, *, capacity_bytes: int | None = None) -> int:
    """Clamp filesystem free bytes to capacity minus current filesystem usage."""
    if capacity_bytes is not None and (
            isinstance(capacity_bytes, bool) or not isinstance(capacity_bytes, int)
            or capacity_bytes <= 0):
        raise ValueError('capacity_bytes must be a positive integer or None')
    usage = shutil.disk_usage(path)
    if capacity_bytes is None:
        return usage.free
    return min(usage.free, max(0, capacity_bytes - (usage.total - usage.free)))


def decimal_gb_bytes(value: int | float) -> int | None:
    """Convert decimal GB to whole bytes; zero disables the capacity ceiling."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('disk capacity GB must be a finite nonnegative number')
    amount = Decimal(str(value))
    if not amount.is_finite() or amount < 0:
        raise ValueError('disk capacity GB must be a finite nonnegative number')
    if amount == 0:
        return None
    numerator, denominator = amount.as_integer_ratio()
    capacity = numerator * 1_000_000_000 // denominator
    if capacity < 1:
        raise ValueError('positive disk capacity must represent at least one byte')
    return capacity
