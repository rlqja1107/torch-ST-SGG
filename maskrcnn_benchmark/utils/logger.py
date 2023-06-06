# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys
from termcolor import colored
import shutil
from tensorboardX import SummaryWriter
from maskrcnn_benchmark.utils.comm import get_rank
from maskrcnn_benchmark.utils.metric_logger import SmoothedValue

DEBUG_PRINT_ON = True

def debug_print(logger, info):
    if DEBUG_PRINT_ON:
        logger.info('#'*20+' '+info+' '+'#'*20)

class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    logger = logging.getLogger(name)

    for each in logger.handlers:
        logger.removeHandler(each)

    logger.setLevel(4)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter =  _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
            )
    ch.setFormatter(formatter)
    logger.addHandler(ch)


    if save_dir:

        tf = TFBoardHandler(TFBoardWriter(save_dir))
        tf.setLevel(4)
        logger.addHandler(tf)

        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

class TFBoardHandler(logging.Handler):
    def __init__(self, writer):
        logging.Handler.__init__(self, 4)
        self.tf_writer = writer

    def emit(self, record):
        if record.levelno <= 4:
            self.tf_writer.write_data(record.msg[0], record.msg[1])
        return

    def close(self):
        self.tf_writer.close()
        
        

class TFBoardWriter:
    def __init__(self, log_dir):
        if log_dir and get_rank() == 0:
            tfbd_dir = os.path.join(log_dir, 'tfboard')
            if os.path.exists(tfbd_dir):
                shutil.rmtree(tfbd_dir)
            os.makedirs(tfbd_dir)

            self.tf_writer = SummaryWriter(log_dir=tfbd_dir, flush_secs=10)
            self.enable = True
        else:
            self.enable = False
            self.tf_writer = None

    def write_data(self, meter, iter):
        if isinstance(iter, str):
            model = meter[0]
            input = meter[1]

            self.tf_writer.add_graph(model, input)
        else:
            for each in meter.keys():
                val = meter[each]
                if isinstance(val, SmoothedValue):
                    val = val.avg
                self.tf_writer.add_scalar(each, val, iter)

    def close(self):
        if self.tf_writer is not None:
            self.tf_writer.close()
