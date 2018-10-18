from tensorboardX import SummaryWriter
import datetime


class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir, prefix=''):
        """Initialize summary writer."""
        exp_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.writer = SummaryWriter(log_dir+'/'+ prefix + '_' + exp_string)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        self.writer.add_scalar(tag, value, step)
        # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        # self.writer.add_summary(summary, step)