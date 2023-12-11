from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# Initialize the EarlyStopping callback

early_stop_callback_exp_rate = EarlyStopping(
    monitor='val_exp_rate',  # Replace with your validation accuracy metric name
    patience=10,  # Number of epochs with no improvement after which training will be stopped
    strict=False,
    verbose=False,
    mode='max'  # 'max' because we want to monitor an accuracy metric which should be maximized
)

early_stop_callback_loss = EarlyStopping(
    monitor='val_loss',  # Replace with your validation accuracy metric name
    patience=10,  # Number of epochs with no improvement after which training will be stopped
    strict=False,
    verbose=False,
    mode='min'  # 'max' because we want to monitor an accuracy metric which should be maximized
)

# Set up the checkpoint callback
# Checkpoint to save the best model based on maximum validation expression rate
checkpoint_callback_exp_rate_3 = ModelCheckpoint(
    dirpath='checkpoints',
    filename='best-checkpoint-exp-rate',
    save_top_k=1,
    verbose=True,
    monitor='val_exp_rate_less_3',
    mode='max'
)

# Checkpoint to save the best model based on minimum validation loss
checkpoint_callback_val_loss = ModelCheckpoint(
    dirpath='checkpoints',
    filename='best-checkpoint-val-loss',
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min'
)