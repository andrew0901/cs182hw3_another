from argparse import Namespace

#######################################################################
# TODO: Design your own neural network
# Set hyperparameters here
#######################################################################
HP = Namespace(
    batch_size=32,
    lr=8e-4,
    momentum=0.85,
    lr_decay=0.97,
    optim_type="adam",
    l2_reg=0.0001,
    epochs=7,
    do_batchnorm=True,
    p_dropout=0.15
)
#######################################################################
