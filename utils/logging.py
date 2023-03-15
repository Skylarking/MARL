from collections import defaultdict
import logging
import numpy as np
from tensorboardX.writer import SummaryWriter


class Logger:
    def __init__(self):

        self.use_tb = False

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboardX import SummaryWriter
        self.writer = SummaryWriter(logdir=directory_name)
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))
        if self.use_tb:
            self.writer.add_scalar(key, value, t)




# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    return logger

# if __name__ == '__main__':
#     import os
#     log = Logger()
#     log.setup_tb('./result')
#     log.log_stat('loss', 10 , 23)

