from lightning.pytorch.callbacks import Callback, ProgressBar


class KstyleBar(ProgressBar):
    def __init__(self, bar_length, args, train_stat = ['v_num'], val_stat = ['v_num'], test_stat = ['v_num']):
        super().__init__()
        self.bar_length = int(bar_length)
        self.epoch_format = args['EPOCHS']

        # Provide the stat you want to track in train, val, and test progress bar
        self.train_stat = set(train_stat)
        self.val_stat = set(val_stat)
        self.test_stat = set(test_stat)


        # recording loss of each step for average
        self.loss_avg_train = {}
        self.loss_avg_val = {}
        self.loss_avg_test = {}

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # Clear out data from previous epoch
        self.loss_avg_train = {}
        self.loss_avg_val = {}
        self.loss_avg_test = {}



    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        epoch_info = 'Epoch[{current_epoch:4.0f}/{max_epoch:4.0f}] '.format(
            current_epoch = trainer.current_epoch, max_epoch = trainer.max_epochs
        )

        # Show the percentage of progress
        percent = (batch_idx / (trainer.num_training_batches - 1))
        percent_info = "{:6.2f}% ".format(percent * 100)

        # Shows the batch index of progress
        batch_idx_info = '{batch_idx:4.0f}/{num_training_batches:4.0f} '.format(
            batch_idx=batch_idx + 1, num_training_batches=trainer.num_training_batches
        )

        bar = list('=' * int(percent * self.bar_length) + '>' + '·' * (self.bar_length - int(percent * self.bar_length)))
        bar[0], bar[-1] = '[', ']'
        bar = ''.join(bar)

        # print(self.get_metrics(trainer, pl_module))


        # Loss info
        loss_info = ''
        for stat, values in self.get_metrics(trainer, pl_module).items():
            if stat[:5] == 'train':
            #if stat in self.train_stat:
                loss_info += '{stat_name}: {stat_value:3.4f} '.format(
                    stat_name=stat, stat_value=values
                )

        print("\r" + epoch_info + percent_info + bar + batch_idx_info + loss_info, end='\r', flush=True)

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print()


    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx: int = 0,
    ):
        percent = (batch_idx / (trainer.num_val_batches[0] - 1))
        percent_info = "{:6.2f}% ".format(percent * 100)
        batch_idx_info = '{batch_idx:4.0f}/{num_training_batches:4.0f} '.format(
            batch_idx=batch_idx + 1, num_training_batches=trainer.num_val_batches[0]
        )

        bar = list('=' * int(percent * self.bar_length) + '>' + '·' * (self.bar_length - int(percent * self.bar_length)))
        bar[0], bar[-1] = '[', ']'
        bar = ''.join(bar)

        # Loss info
        loss_info = ''
        for stat, values in self.get_metrics(trainer, pl_module).items():
            if stat[:5] == 'valid':
                loss_info += '{stat_name}: {stat_value:3.4f} '.format(
                    stat_name=stat, stat_value=values
                )

        print("\r" +'Evaluating... ' + percent_info + bar + batch_idx_info + loss_info, end='\r', flush=True)

