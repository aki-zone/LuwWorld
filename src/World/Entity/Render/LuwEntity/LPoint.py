class LPoint:
    def __init__(self, x:float=0, y:float=0, z:float=None):
        self.x = x
        self.y = y
        self.z = z

    def isNull(self):
        return self.x is None and self.y is None and self.z is None

    def manhattanLength(self):
        if self.z is not None:
            return abs(self.x) + abs(self.y) + abs(self.z)
        else:
            return abs(self.x) + abs(self.y)

    def __eq__(self, other):
        if self.z is not None and other.z is not None:
            return self.x == other.x and self.y == other.y and self.z == other.z
        else:
            return self.x == other.x and self.y == other.y

    def __str__(self):
        if self.z is not None:
            return f'({self.x}, {self.y}, {self.z})'
        else:
            return f'({self.x}, {self.y})'

    def is_2d(self):
        return self.z is None

    def is_3d(self):
        return self.z is not None

    # Getter methods
    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getZ(self):
        return self.z

    # Setter methods
    def setX(self, x:float):
        self.x = x

    def setY(self, y:float):
        self.y = y

    def setZ(self, z:float):
        self.z = z
