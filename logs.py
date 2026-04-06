from torch.utils.tensorboard import SummaryWriter


class CustomTensorBoard:
    def __init__(self, log_dir='logs'):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log(self, step, **stats):
        for name, value in stats.items():
            self.writer.add_scalar(name, value, step)
        self.writer.flush()
