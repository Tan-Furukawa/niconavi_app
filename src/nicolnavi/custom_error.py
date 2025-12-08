class MyException(Exception):
    def __init__(self, arg: str = "") -> None:
        super().__init__(arg)
        self.arg = arg

class NoVideoError(MyException):
    def __str__(self) -> str:
        if self.arg != "":
            return f"No video selected ({self.arg})"
        else:
            return f"No video selected"


class RotatedAngleError(MyException):
    def __str__(self) -> str:
        if self.arg != "":
            return f"{self.arg}"
        else:
            return f"rotation angle is not enough"


class UnexpectedNoneType(MyException):
    def __str__(self) -> str:
        if self.arg != "":
            return f"Unexpected None type ({self.arg})"
        else:
            return f"Unexpected None type"


class InvalidRotationDirection(MyException):
    def __str__(self) -> str:
        if self.arg != "":
            return f"Incorrect rotation direction of {self.arg}."
        else:
            return f"Incorrect rotation direction."
