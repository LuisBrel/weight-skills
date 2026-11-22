"""Exception hierarchy for weight-skills."""

from __future__ import annotations


class WeightSkillsError(Exception):
    """Base class for every weight-skills error."""


class CartridgeFormatError(WeightSkillsError):
    """Raised when a cartridge file is malformed or fails schema validation."""


class GeometryMismatchError(WeightSkillsError):
    """Raised when a cartridge does not match the target model's hidden geometry."""


class BackendError(WeightSkillsError):
    """Raised when no execution backend can serve a model, or a mount fails."""


class RestorationError(WeightSkillsError):
    """Raised when unmounting fails to restore the pristine base weights bit-for-bit."""


class CompilationError(WeightSkillsError):
    """Raised when closed-form cartridge compilation cannot be completed."""
