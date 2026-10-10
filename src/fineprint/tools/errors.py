"""The one error a tool raises when it cannot answer from the handbook."""


class ToolError(ValueError):
    """A tool was asked something it cannot answer: a malformed argument, an unknown figure, or a
    case the handbook does not cover. The message goes back to the model as the tool's result,
    so it is written for the model to read and pass on.
    """
