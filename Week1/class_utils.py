from dataclasses import dataclass


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    track_id: int
    frame: int
    label: str
    parked: bool

    def __post_init__(self):
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1

    def __repr__(self):
        return f'BoundingBox(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2}, frame={self.frame}, track_id={self.track_id}, label={self.label})'

    def __eq__(self, other):
        return (self.x1 == other.x1 and self.y1 == other.y1
                and self.x2 == other.x2 and self.y2 == other.y2
                and self.frame == other.frame
                and self.track_id == other.track_id
                and self.label == other.label
                )

    def __hash__(self):
        return hash((self.x1, self.y1, self.x2, self.y2, self.frame, self.track_id, self.label))

    def clone(self):
        return BoundingBox(self.x1, self.y1, self.x2, self.y2, self.track_id, self.frame, self.label, self.parked)