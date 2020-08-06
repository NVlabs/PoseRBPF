# ----------------------------------------------------------------------------
# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# ----------------------------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import datasets.ycb_object
import numpy as np

# ycb object dataset
for split in ['train', 'test']:
    name = 'ycb_object_{}'.format(split)
    print(name)
    __sets[name] = (lambda split=split:
            datasets.YCBObject(split))


def get_dataset(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_datasets():
    """List all registered imdbs."""
    return __sets.keys()
