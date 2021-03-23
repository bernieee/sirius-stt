from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    def __init__(self, tensorboard_path):
        self.writer = SummaryWriter(tensorboard_path)

    def log(self, step, loss, wer, mode):
        # write your code here

        # add loss to tb
        # add wer to tb
        self.writer.add_scalar(f'wer/{mode}', wer, global_step=step)
        self.writer.add_scalar(f'loss/{mode}', loss, global_step=step)

    def log_text(self, step, pred_texts, gt_texts, mode):

        # write your code here

        for pred_text in pred_texts:
            # add pred text to tb
            self.writer.add_text(f'predict/{mode}', pred_text, global_step=step)

        for gt_text in gt_texts:
            # add gt text to tb
            self.writer.add_text(f'ground truth/{mode}', gt_text, global_step=step)

    def close(self):
        self.writer.close()
