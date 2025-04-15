from dataclasses import dataclass

@dataclass(frozen=True)
class CarState:
    row: int
    col: int
    v_row: int
    v_col: int

    def position(self):
        return self.row, self.col

    def velocity(self):
        return self.v_row, self.v_col