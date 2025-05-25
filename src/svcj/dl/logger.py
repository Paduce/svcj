import logging
from torch.utils.tensorboard import SummaryWriter

class TBLogger:
    def __init__(self, log_dir: str, log_to_file: bool = False, log_file: str = "train.log", log_level: int = logging.INFO):
        # TensorBoard writer
        self.w = SummaryWriter(log_dir)

        # Python logger
        self.logger = logging.getLogger("TBLogger")
        self.logger.setLevel(log_level)
        self.logger.propagate = False  # Avoid duplicate logs

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # Optional file handler
        if log_to_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(log_level)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def log(self, step: int, value: float, tag: str = "train/loss", verbose: bool = True):
        # TensorBoard log
        self.w.add_scalar(tag, value, step)
        # Standard log
        if verbose:
            self.logger.info(f"{tag} @ step {step}: {value:.6f}")

    def log_msg(self, msg: str, level: int = logging.INFO):
        # General-purpose message logging
        self.logger.log(level, msg)

    def close(self):
        self.w.close()
        # Close all handlers (especially important if logging to file)
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
