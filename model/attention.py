import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__()
        # Linear transformations for the attention mechanism
        self.encoder_att = nn.Linear(encoder_dim, decoder_dim)
        self.decoder_att = nn.Linear(decoder_dim, decoder_dim)
        self.full_att = nn.Linear(decoder_dim, 1)

        # Activation and normalization functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # Compute attention scores and weights
        att1 = self.encoder_att(encoder_out)  # Transform encoder output
        att2 = self.decoder_att(decoder_hidden)  # Transform decoder hidden state
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # Combine and apply non-linearity
        alpha = self.softmax(att)  # Calculate attention weights
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # Weighted sum of encoder outputs

        return attention_weighted_encoding, alpha


class AttentionDecoder(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=256, dropout=0.5):
        super().__init__()
        # Dimensions
        self.embed_dim = embed_dim
        self.encoder_dim = encoder_dim
        self.vocab_size = vocab_size

        # Components of the attention decoder
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # Word embedding layer
        self.attention = Attention(encoder_dim, decoder_dim)  # Attention mechanism
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # LSTMCell for decoding steps
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # Linear layer to find initial hidden state of LSTM
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # Linear layer to find initial cell state of LSTM
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # Linear layer to create a gating scalar
        self.fc = nn.Linear(decoder_dim, vocab_size)  # Final linear layer to predict the next word
        self.dropout = nn.Dropout(dropout)  # Dropout layer

        self.init_weights()  # Initialize weights

    def init_weights(self):
        # Initialize weights for stable training
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        max_len = max(caption_lengths)  # Find the maximum sequence length in this batch

        # Initialize tensors
        predictions = torch.zeros(batch_size, max_len, self.vocab_size).to(encoder_out.device)
        embeddings = self.embedding(encoded_captions)  # Get embeddings for encoded captions

        # Prepare the initial LSTM hidden and cell states
        h, c = self.init_hidden_state(encoder_out)

        # Inside the AttentionDecoder forward loop:
        alphas = torch.zeros(batch_size, max_len, encoder_out.size(1)).to(encoder_out.device)

        # Sequentially generate the output tokens
        for t in range(max_len):
            batch_size_t = sum([l > t for l in caption_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = torch.sigmoid(self.f_beta(h[:batch_size_t]))  # Compute gating scalar
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                                    (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout(h))  # Compute predictions
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, caption_lengths, alphas

    def init_hidden_state(self, encoder_out):
        # Compute the mean of encoder output to initialize LSTM hidden and cell states
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c
