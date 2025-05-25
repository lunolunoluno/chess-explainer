class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Debug(metaclass=SingletonMeta):
    def __init__(self, debug=False):
        self.debug = debug

    def set_debug(self, value: bool):
        self.debug = value

    def print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)