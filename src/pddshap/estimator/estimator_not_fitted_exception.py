class EstimatorNotFittedException(Exception):
    def __init__(self) -> None:
        super().__init__("Must call fit() before using __call__()")
