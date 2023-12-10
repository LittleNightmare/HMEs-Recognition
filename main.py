from argparse import ArgumentParser

from lightning import Trainer

from callbacks import checkpoint_callback_exp_rate_3, checkpoint_callback_val_loss, early_stop_callback_exp_rate, \
    early_stop_callback_loss
from dataset.data_module import MathExpressionDataModule
from model.HMERecognizer import HMERecognizer
from utils import decode_truth, END

parser = ArgumentParser()

# DataModule arguments
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=8)
# Trainer arguments
parser.add_argument("--max_epochs", type=int, default=100)

# Hyperparameters for the model
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--encoder_out_dim", type=int, default=512)

# Callbacks
parser.add_argument("--early_stop", action='store_true')
parser.add_argument("--checkpoint", action='store_true')

# Other arguments
parser.add_argument("--train", action='store_true')
parser.add_argument("--test", action='store_true')
parser.add_argument("--sample", action='store_true')
parser.add_argument("--checkpoint_path", type=str, default="checkpoints/best-checkpoint-exp-rate.ckpt")


def main(args):
    data_module = MathExpressionDataModule(data_dir=args.data_dir, tokens_file='tokens.tsv', batch_size=args.batch_size,
                                           num_workers=args.num_workers)

    model = HMERecognizer(token_to_id=data_module.token_to_id, lr=args.lr, encoder_out_dim=args.encoder_out_dim,
                          vocab_size=len(data_module.token_to_id), batch_size=args.batch_size)

    callbacks = []
    if args.early_stop:
        callbacks.append(early_stop_callback_exp_rate)
        callbacks.append(early_stop_callback_loss)
    if args.checkpoint:
        callbacks.append(checkpoint_callback_exp_rate_3)
        callbacks.append(checkpoint_callback_val_loss)

    if args.train:
        trainer = Trainer(max_epochs=args.max_epochs, callbacks=callbacks)
        data_module.setup('train')
        trainer.fit(model, datamodule=data_module)
    if args.test:
        trainer = Trainer(max_epochs=args.max_epochs, callbacks=callbacks)
        data_module.setup('test')
        model = HMERecognizer.load_from_checkpoint(checkpoint_path=args.checkpoint_path,
                                                   token_to_id=data_module.token_to_id, lr=args.lr,
                                                   encoder_out_dim=args.encoder_out_dim,
                                                   vocab_size=len(data_module.token_to_id))
        results = {}
        for year, dataloader in data_module.test_dataloader().items():
            test_dataloader = dataloader

            # Run the test set through the trained model
            results[year] = trainer.test(model, dataloaders=test_dataloader)
        # generate confusion matrix based on results

    # generate 3 samples from the test set, and print them
    if args.sample:
        # checkpoint = torch.load(args.checkpoint_path)
        # print("Loading model from checkpoint")
        # print(checkpoint["state_dict"])
        data_module.setup('test')
        model = HMERecognizer.load_from_checkpoint(checkpoint_path=args.checkpoint_path,
                                                   token_to_id=data_module.token_to_id, lr=args.lr,
                                                   encoder_out_dim=args.encoder_out_dim,
                                                   vocab_size=len(data_module.token_to_id))
        model.eval()
        trainer = Trainer(fast_dev_run=1)
        for year, dataloader in data_module.test_dataloader().items():
            print(f"Sample on {year} dataset")
            result = trainer.predict(model, dataloaders=dataloader)
            samples = result[0]
            for sample in result:
                samples.extend(sample)
            for i, sample in enumerate(samples):
                if i == 3:
                    break
                image = sample['image']
                truth = sample['truth']['text']
                pred = sample['pred']['text']
                print(f"Sample {i}")
                print(f"  Truth: {decode_truth(truth, data_module.id_to_token, data_module.token_to_id[END])}")
                print(f"  Pred:  {decode_truth(pred, data_module.id_to_token, data_module.token_to_id[END])}")
                print(f"  Image: {image.shape}")


if __name__ == '__main__':
    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()

    main(args)
