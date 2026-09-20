"""Exception types raised across the nagcsu package.

Catching :class:`NagcsuError` from calling code catches everything this
package raises deliberately; the more specific subclasses let a caller
distinguish a bad deck edit from a failed simulation from a scoring
problem without parsing error strings.
"""


class NagcsuError(Exception):
    """Base class for every exception raised deliberately by nagcsu."""


class DeckPatchError(NagcsuError):
    """A parameter edit could not be applied to a deck.

    Raised when a keyword block a :class:`~nagcsu.parameters.ParameterSpec`
    expects to find is missing, or when a patch pattern matches zero or
    more than one place in the deck text (an ambiguous edit is refused
    rather than guessed at).
    """


class SimulationError(NagcsuError):
    """An OPM Flow run failed to launch or exited with a nonzero status.

    Raised by :mod:`nagcsu.simulate`. The wrapped :attr:`stderr` and
    :attr:`returncode` are attached so a caller can print or log the
    underlying OPM Flow diagnostics without re-reading the run directory.
    """

    def __init__(self, message: str, *, returncode: int | None = None, stderr: str = "") -> None:
        super().__init__(message)
        self.returncode = returncode
        """Process exit code of the failed `flow` invocation, if known."""
        self.stderr = stderr
        """Captured standard error output of the failed `flow` invocation."""


class RunOutputNotFoundError(NagcsuError):
    """No summary output was found in a run directory after a simulation.

    Raised when a run directory does not contain the UNSMRY/SMSPEC pair
    OPM Flow is expected to have written, whatever their basename turned
    out to be (see :func:`nagcsu.simulate.find_case_basename`).
    """


class HistoryAlignmentError(NagcsuError):
    """The observed history and a simulated summary could not be aligned.

    Raised by :mod:`nagcsu.objective` when the two frames share no
    common dates or no common scored vectors, which usually means the
    history workbook's column mapping in the project config is wrong.
    """
