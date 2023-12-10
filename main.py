
from lightning import Trainer

from dataset.data_module import MathExpressionDataModule
from model.HMERecognizer import HMERecognizer
from argparse import ArgumentParser

from callbacks import checkpoint_callback_exp_rate_3, checkpoint_callback_val_loss, early_stop_callback_exp_rate, \
    early_stop_callback_loss

parser = ArgumentParser()

# DataModule arguments
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=8)
# Trainer arguments
parser.add_argument("--max_epochs", type=int, default=100)


# Hyperparameters for the model
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--encoder_out_dim", type=int, default=512)

# Callbacks
parser.add_argument("--early_stop", action='store_true')
parser.add_argument("--checkpoint", action='store_true')


# Other arguments
parser.add_argument("--train", action='store_true')
parser.add_argument("--checkpoint_path", type=str, default="checkpoints/best-checkpoint-exp-rate.ckpt")

# Parse the user inputs and defaults (returns a argparse.Namespace)
args = parser.parse_args()

data_module = MathExpressionDataModule(data_dir=args.data_dir, tokens_file='tokens.tsv', batch_size=args.batch_size, num_workers=args.num_workers)

model = HMERecognizer(token_to_id=data_module.token_to_id, lr=args.lr, encoder_out_dim=args.encoder_out_dim, vocab_size=len(data_module.token_to_id), batch_size=args.batch_size)

callbacks = []
if args.early_stop:
    callbacks.append(early_stop_callback_exp_rate)
    callbacks.append(early_stop_callback_loss)
if args.checkpoint:
    callbacks.append(checkpoint_callback_exp_rate_3)
    callbacks.append(checkpoint_callback_val_loss)

trainer = Trainer(max_epochs=args.max_epochs, callbacks=callbacks)

if args.train:
    trainer.fit(model, datamodule=data_module)
else:
    data_module.setup('test')
    model = HMERecognizer.load_from_checkpoint(checkpoint_path=args.checkpoint_path)
    model.eval()
    for dataset_name, dataloader in data_module.test_dataloader():
        print(f"Testing on {dataset_name} dataset")
        trainer.test(model, dataloaders=dataloader)