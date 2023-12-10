import csv

import torch

START = "<SOS>"
END = "<EOS>"
PAD = "<PAD>"
SPECIAL_TOKENS = [START, END, PAD]


def remove_unknown_tokens(truth):
    # Remove \mathrm and \vtop are only present in the test sets, but not in the
    # training set. They are purely for formatting anyway.
    remaining_truth = truth.replace("\\mathrm", "")
    remaining_truth = remaining_truth.replace("\\vtop", "")
    # \; \! are spaces and only present in 2014's test set
    remaining_truth = remaining_truth.replace("\\;", " ")
    remaining_truth = remaining_truth.replace("\\!", " ")
    remaining_truth = remaining_truth.replace("\\ ", " ")
    # There's one occurrence of \dots in the 2013 test set, but it wasn't present in the
    # training set. It's either \ldots or \cdots in math mode, which are essentially
    # equivalent.
    remaining_truth = remaining_truth.replace("\\dots", "\\ldots")
    # Again, \lbrack and \rbrack where not present in the training set, but they render
    # similar to \left[ and \right] respectively.
    remaining_truth = remaining_truth.replace("\\lbrack", "\\left[")
    remaining_truth = remaining_truth.replace("\\rbrack", "\\right]")
    # Same story, where \mbox = \leavemode\hbox
    remaining_truth = remaining_truth.replace("\\hbox", "\\mbox")
    # There is no reason to use \lt or \gt instead of < and > in math mode. But the
    # training set does. They are not even LaTeX control sequences but are used in
    # MathJax (to prevent code injection).
    remaining_truth = remaining_truth.replace("<", "\\lt")
    remaining_truth = remaining_truth.replace(">", "\\gt")
    # \parallel renders to two vertical bars
    remaining_truth = remaining_truth.replace("\\parallel", "||")
    # Some capital letters are not in the training set...
    remaining_truth = remaining_truth.replace("O", "o")
    remaining_truth = remaining_truth.replace("W", "w")
    remaining_truth = remaining_truth.replace("\\Pi", "\\pi")
    return remaining_truth


# Rather ignorant way to encode the truth, but at least it works.
def encode_truth(truth, token_to_id):
    truth_tokens = []
    remaining_truth = remove_unknown_tokens(truth).strip()
    while len(remaining_truth) > 0:
        try:
            matching_starts = [
                [i, len(tok)]
                for tok, i in token_to_id.items()
                if remaining_truth.startswith(tok)
            ]
            # Take the longest match
            index, tok_len = max(matching_starts, key=lambda match: match[1])
            truth_tokens.append(index)
            remaining_truth = remaining_truth[tok_len:].lstrip()
        except ValueError:
            raise Exception("Truth contains unknown token")
    return truth_tokens


def load_vocab(tokens_file):
    with open(tokens_file, "r") as fd:
        reader = csv.reader(fd, delimiter="\t")
        tokens = next(reader)
        tokens.extend(SPECIAL_TOKENS)
        token_to_id = {tok: i for i, tok in enumerate(tokens)}
        id_to_token = {i: tok for i, tok in enumerate(tokens)}
        return token_to_id, id_to_token


def collate_batch(data, pad_index):
    # Find the length of the longest sequence
    max_len = max(d['truth']['length'] for d in data)

    # Pad all sequences to match the length of the longest one
    padded_encoded = torch.full((len(data), max_len), pad_index, dtype=torch.long)
    for i, d in enumerate(data):
        length = d['truth']['length']
        padded_encoded[i, :length] = d['truth']['encoded']

    # Stack all images into a single tensor
    images = torch.stack([d['image'] for d in data], dim=0)

    # Also, collect the lengths of all sequences
    lengths = torch.tensor([d['truth']['length'] for d in data], dtype=torch.long)

    return {
        'image': images,
        'truth': {
            'text': [d['truth']['text'] for d in data],
            'encoded': padded_encoded,
            'length': lengths  # The lengths of each sequence
        },
    }


def decode_truth(encoded_truth, id_to_token, end_token):
    encoded_truth = encoded_truth.tolist()
    for i, token in enumerate(encoded_truth):
        if token == end_token:
            encoded_truth = encoded_truth[1:i]
            break
    return "".join([id_to_token[i] for i in encoded_truth])
