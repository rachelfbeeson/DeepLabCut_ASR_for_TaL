from tools.UtteranceController import UtteranceController
from tools.config_manager import config


class MainSetup:
    def __init__(self):
        tal_setup = UtteranceController(config.getboolean('Run', 'make_features'))
        tal_setup.forward()

if __name__ == '__main__':
    MainSetup()
