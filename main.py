from controller import Controller
from modules.utils import Debug

if __name__ == "__main__":
    dbg = Debug(debug=True)
    dbg.print("Debug: On")

    ctrl = Controller()
    ctrl.update_dataset()
