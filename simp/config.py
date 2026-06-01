from dataclasses import dataclass


@dataclass
class SimpConfig:
    nelx: int = 100
    nely: int = 100
    volfrac: float = 0.4
    penal: float = 3.0
    rmin: float = 3.0
    ft: int = 2
    objective_type: str = 'first'
    max_iter: int = 200
    move: float = 0.2
    save_every: int = 1
    scale_factor: int = 1
    beta: float = 0.85
    beta_second: float = 1.0
    E0: float = 199.0
    Emin: float = 1e-9
    nu: float = 0.3

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        assert self.nelx > 0, 'nelx must be positive'
        assert self.nely > 0, 'nely must be positive'
        assert 0 < self.volfrac <= 1, 'volfrac must be in (0, 1]'
        assert self.penal >= 1, 'penal must be >= 1'
        assert self.rmin > 0, 'rmin must be positive'
        assert self.ft in (1, 2), 'ft must be 1 or 2'
        assert self.objective_type in ('first', 'second', 'auxetic'), (
            'objective_type must be first, second, or auxetic'
        )
        assert self.max_iter > 0, 'max_iter must be positive'
        assert 0 < self.move <= 1, 'move must be in (0, 1]'
        assert self.save_every > 0, 'save_every must be positive'
        assert self.scale_factor >= 1, 'scale_factor must be >= 1'
