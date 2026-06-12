from highway_env.object.abc import RoadObject


__all__ = ["Obstacle", "Landmark"]


class Obstacle(RoadObject):
    """Obstacles on the road."""

    def __init__(
        self, road, position, heading: float = 0, speed: float = 0
    ):
        super().__init__(road, position, heading, speed)
        self.solid: bool = True


class Landmark(RoadObject):
    """Landmarks of certain areas on the road that must be reached."""

    def __init__(
        self, road, position, heading: float = 0, speed: float = 0
    ):
        super().__init__(road, position, heading, speed)
        self.solid: bool = False
