from typing import Optional

from pytorch_lightning.callbacks import Callback


class ModelExport(Callback):
    def __init__(self, dirpath: str = "", filename: Optional[str]=None, period: int=1):
        super().__init__()
        self.period = period
        self.dirpath = dirpath
        self.filename = filename
        if self.filename is None:
            self.filename = "model_{0}.onnx"

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.period == 0:
            filepath = self.dirpath + "/" + self.filename.format(trainer.current_epoch)
            pl_module.export(filepath)
