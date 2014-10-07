import logging

def get_logger(name=None, level=logging.DEBUG, fpath=None, console=True):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] %(message)s')
        if fpath is not None:
            fh = logging.FileHandler(fpath)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        if console:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            logger.addHandler(sh)
    return logger
