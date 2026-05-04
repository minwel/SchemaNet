class TimeoutException(Exception):
    def __init__(self,):
        super().__init__()

    def __str__(self) -> str:
        return "this code runs too long"
