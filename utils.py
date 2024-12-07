import os
import logging
import time
import errno
import torch

def makedirsExist(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory not created.')
        else:
            raise


def create_logger(output_path):
    # set up logger
    if not os.path.exists(output_path):
        makedirsExist(output_path)
    assert os.path.exists(output_path), '{} does not exist'.format(output_path)

    log_file = 'log_{}_{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'), output_path.split('/')[1])
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(output_path, log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger


def time_batch(gen, reset_every=100):
    """
    Gets timing info for a batch
    :param gen:
    :param reset_every: How often we'll reset
    :return:
    """
    start = time.time()
    start_t = 0
    for i, item in enumerate(gen):
        time_per_batch = (time.time() - start) / (i + 1 - start_t)
        yield time_per_batch, item
        if i % reset_every == 0:
            start = time.time()
            start_t = i


def to_cuda(batch):
    batch = list(batch)

    for i in range(len(batch)):
        if isinstance(batch[i], torch.Tensor):
            batch[i] = batch[i].cuda(non_blocking=True)
        elif isinstance(batch[i], list):
            for j, o in enumerate(batch[i]):
                if isinstance(batch[i], torch.Tensor):
                    batch[i][j] = o.cuda(non_blocking=True)

    return batch