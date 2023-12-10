import lightning as pl
import torchmetrics
import torch
from torch import nn

from model.attention import AttentionDecoder
from model.encoder import CNNEncoder, RowEncoder


class HMERecognizer(pl.LightningModule):
    def __init__(self, token_to_id, lr=0.1, encoder_out_dim=512, vocab_size=119, batch_size=32):
        super().__init__()
        self.token_to_id = token_to_id
        self.lr = lr
        self.encoder_out_dim = encoder_out_dim
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        # Define the network components as before
        self.encoder = CNNEncoder()
        self.row_encoder = RowEncoder(feature_dim=self.encoder_out_dim * 32, hidden_dim=256)
        # self.attention = Attention(encoder_dim=2 * 256, decoder_dim=512)
        self.decoder = AttentionDecoder(attention_dim=256, embed_dim=256, decoder_dim=512, vocab_size=self.vocab_size,
                                        encoder_dim=2 * 256)
        # self.loss_function = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is your padding index
        self.loss_function = nn.CrossEntropyLoss()

        # Initialize torchmetrics
        self.accuracy_metric = torchmetrics.Accuracy(num_classes=vocab_size, average='macro', task='multiclass')
        self.precision_metric = torchmetrics.Precision(num_classes=vocab_size, average='macro', task='multiclass')
        self.recall_metric = torchmetrics.Recall(num_classes=vocab_size, average='macro', task='multiclass')
        self.f1_metric = torchmetrics.F1Score(num_classes=vocab_size, average='macro', task='multiclass')
        # self.confusion_matrix_metric = torchmetrics.ConfusionMatrix(num_classes=vocab_size, normalize='all',
        #                                                             task='multiclass')

    def forward(self, images, equations, equation_lengths):
        features = self.encoder(images)
        row_features = self.row_encoder(features)
        predictions, _, _ = self.decoder(row_features, equations, equation_lengths)
        return predictions

    def training_step(self, batch, batch_idx):
        images = batch['image']
        encoded_captions = batch['truth']['encoded']
        caption_lengths = batch['truth']['length']

        output = self(images, encoded_captions, caption_lengths)
        loss = self.loss_function(output.view(-1, self.vocab_size), encoded_captions.view(-1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size)
        return loss

    def calc_exp_rate(self, preds, targets):
        # Initialize mask for <PAD>, <SOS>, and <EOS> tokens
        mask = (targets != self.token_to_id['<PAD>']) & \
               (targets != self.token_to_id['<SOS>']) & \
               (targets != self.token_to_id['<EOS>'])

        # Calculate expression rate with different error tolerances
        num_correct_tokens = torch.sum((preds == targets) & mask, dim=1)
        num_tokens = torch.sum(mask, dim=1)
        errors = num_tokens - num_correct_tokens

        exp_rate = (errors == 0).float().mean()
        exp_rate_less_1 = (errors <= 1).float().mean()
        exp_rate_less_2 = (errors <= 2).float().mean()
        exp_rate_less_3 = (errors <= 3).float().mean()

        return exp_rate, exp_rate_less_1, exp_rate_less_2, exp_rate_less_3

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        encoded_captions = batch['truth']['encoded']
        caption_lengths = batch['truth']['length']

        # Forward pass
        output = self(images, encoded_captions, caption_lengths)
        loss = self.loss_function(output.view(-1, self.vocab_size), encoded_captions.view(-1))

        # Calculate ExpRate
        preds = torch.argmax(output, dim=2)  # Get the index of the max log-probability
        # preds = preds.cpu()
        targets = encoded_captions
        self.update_metrics(preds, targets)
        exp_rate, exp_rate_less_1, exp_rate_less_2, exp_rate_less_3 = self.calc_exp_rate(preds, targets)

        # Log metrics
        self.log('val_loss', loss, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log('val_exp_rate', exp_rate, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log('val_exp_rate_less_1', exp_rate_less_1, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log('val_exp_rate_less_2', exp_rate_less_2, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log('val_exp_rate_less_3', exp_rate_less_3, prog_bar=True, logger=True, batch_size=self.batch_size)

        return {
            'val_loss': loss,
            'val_exp_rate': exp_rate,
            'val_exp_rate_less_1': exp_rate_less_1,
            'val_exp_rate_less_2': exp_rate_less_2,
            'val_exp_rate_less_3': exp_rate_less_3
        }

    def test_step(self, batch, batch_idx):
        images = batch['image']
        encoded_captions = batch['truth']['encoded']
        caption_lengths = batch['truth']['length']

        # Forward pass
        output = self(images, encoded_captions, caption_lengths)

        # Calculate ExpRate
        preds = torch.argmax(output, dim=2)  # Get the index of the max log-probability
        # preds = preds.cpu()
        targets = encoded_captions
        self.update_metrics(preds, targets)
        exp_rate, exp_rate_less_1, exp_rate_less_2, exp_rate_less_3 = self.calc_exp_rate(preds, targets)

        # Log metrics
        self.log('test_exp_rate', exp_rate, prog_bar=True, batch_size=self.batch_size)
        self.log('test_exp_rate_less_1', exp_rate_less_1, prog_bar=True, batch_size=self.batch_size)
        self.log('test_exp_rate_less_2', exp_rate_less_2, prog_bar=True, batch_size=self.batch_size)
        self.log('test_exp_rate_less_3', exp_rate_less_3, prog_bar=True, batch_size=self.batch_size)

        return {
            'test_exp_rate': exp_rate,
            'test_exp_rate_less_1': exp_rate_less_1,
            'test_exp_rate_less_2': exp_rate_less_2,
            'test_exp_rate_less_3': exp_rate_less_3
        }

    def update_metrics(self, preds, targets):
        # Update torchmetrics
        self.accuracy_metric(preds, targets)
        self.precision_metric(preds, targets)
        self.recall_metric(preds, targets)
        self.f1_metric(preds, targets)
        # self.confusion_matrix_metric(preds, targets)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch['image']
        encoded_captions = batch['truth']['encoded']
        caption_lengths = batch['truth']['length']
        output = self(images, encoded_captions, caption_lengths)
        preds = torch.argmax(output, dim=2)
        preds = preds.cpu()
        targets = encoded_captions.cpu()

        result = []
        for i in range(len(images)):
            result.append({
                'image': images[i],
                'truth': {
                    'text': encoded_captions[i],
                    'encoded': encoded_captions[i],
                    'length': caption_lengths[i]
                },
                'pred': {
                    'text': preds[i],
                    'encoded': preds[i],
                    'length': caption_lengths[i]
                }
            })
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def _epoch_end(self):
        accuracy = self.accuracy_metric.compute()
        precision = self.precision_metric.compute()
        recall = self.recall_metric.compute()
        f1 = self.f1_metric.compute()
        # confusion_matrix = self.confusion_matrix_metric.compute()
        # Log metrics at the end of each epoch
        self.log('accuracy', accuracy, batch_size=self.batch_size)
        self.log('precision', precision, batch_size=self.batch_size)
        self.log('recall', recall, batch_size=self.batch_size)
        self.log('f1_score', f1, batch_size=self.batch_size)
        # self.log('confusion_matrix', confusion_matrix, batch_size=self.batch_size)

        # Reset metrics for the next epoch
        self.accuracy_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1_metric.reset()
        # self.confusion_matrix_metric.reset()

    def on_test_epoch_end(self):
        self._epoch_end()

    def on_validation_epoch_end(self):
        self._epoch_end()
