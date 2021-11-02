#!/usr/bin/env python3
import os

import multiprocessing as mp
from multiprocessing import Pool
from pprint import pprint
import signal
import subprocess

import numpy as np

import xfr
from xfr import utils
from xfr.utils import iterate_param_sets
from xfr.utils import prune_unneeded_exports
from xfr.utils import normalize_gpus
from xfr.inpainting_game.net_mate_nonmate_dists import calc_mate_nonmate_dists

from xfr import xfr_root
from create_wbnet import create_wbnet

def run_experiment(params, params_export, gpu_queue):
    import torch

    gpu_id = gpu_queue.get()
    process = None
    def try_block():
        # taa: This is a physical GPU ID, as generated by normalize_gpus() call
        # in run_experiments.
        if torch.cuda.is_available():
            device = torch.device("cuda:{}".format(gpu_id))
            print("Running on {}".format(device))
        else:
            print("Running on CPU")
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        params['EXPER_DIR'] = os.path.join(
            xfr_root,
            'output',
            'ROC_Curve_Analysis_Inpainting_Game/Net=%s' % params['NET'][0])

        output_dir = os.path.join(
            os.path.dirname(__file__),
            'generated',
            ('{EXPER_DIR}'
            ).format(**params))

        seed = params['RANDOM_SEED'][0] * 1000

        num_subjects = params['NUM_SUBJECTS']

        net_name = params['NET'][0]

        net = create_wbnet(net_name, device=device)
        npfile = os.path.join(
            output_dir,
            'dists_net=%s_seed=%s.npz' % (net_name, seed))

        if (not params['overwrite'] and
                os.path.exists(npfile)
        ):
            print("\nNot recalculating / overwriting %s. Use '--overwrite' to "
                  "force.\n" % npfile)
            return True

        mate_dists, nonmate_dists = calc_mate_nonmate_dists(
            net, num_subjects, seed, output_dir,
            ijbc_path=params['ijbc_path'],
        )

        np.savez_compressed(
            npfile,
            mate_dists=mate_dists,
            nonmate_dists=nonmate_dists,
        )

    if params['debug']:
        # Pass errors through if debugging
        try:
            try_block()
        finally:
            gpu_queue.put(gpu_id)

    else:
        try:
            try_block()
        except TypeError as e:
            print("\n\n ERROR detected. The parameters are:")
            pprint(params)
        except Exception as e:
            print("ERROR: {}".format(e))
        finally:
            gpu_queue.put(gpu_id)

    return True

def run_experiments(params):
    # Turns into the list of physical GPU IDs that we're requesting,
    # and sets CUDA_VISIBLE_DEVICES to match that.
    # Now: we want to set CUDA_VISIBLE_DEVICES to a single GPU downstream,
    # and that must refer to a physical ID (as in newGpus), not to a logical
    # ID (which is what we're taking the gpus parameter to be).
    # We put the physical IDs from newGpus into the gpu_queue,
    # and let the recipient set CUDA_VISIBLE_DEVICES appropriately.
    oldGpus = params['gpus']
    newGpus = normalize_gpus(params['gpus'], True)
    m = mp.Manager()
    gpu_queue = m.Queue()

    for idx, gpu_id in enumerate(newGpus):
        print('Queueing GPU resource %d (%d)' % (int(gpu_id),
                                                 int(oldGpus[idx])))
        gpu_queue.put(gpu_id)
    req_scale = (
        lambda params:
        params['MASKS_CONSTRUCTION'][0].lower() not in [
            'bbnet_mean_ebp', 'facial_regions'])

    params_export = [
        'NET',
        'RANDOM_SEED',
        'gpus',
    ]
    if not params['debug']:
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        pool = Pool(len(params['gpus']))
        signal.signal(signal.SIGINT, original_sigint_handler)
        try:
            for params_ in iterate_param_sets(params, params_export):
                params_export_ = prune_unneeded_exports(
                    params_export, params_)
                ret = pool.apply_async(
                    run_experiment,
                    args=(params_, params_export_),
                    kwds={'gpu_queue':gpu_queue},
                )
            ret.get(999999999)
        except KeyboardInterrupt:
            print('Caught Keyboard interrupt signal.')
            pool.terminate()
            pool.join()
        else:
            print('Finishing multiprocessing ...')
            pool.close()
            print('Finished multiprocessing normally.')
        pool.join()
        print('Joined pool')
    else:
        for params_ in iterate_param_sets(params, params_export):
            params_export_ = prune_unneeded_exports(
                params_export, params_)
            run_experiment(params_, params_export_, gpu_queue)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        'Example script for calculating embedding distances between subjects. '
        'Used to calculate the matching threshold for STR ResNet v4 and v6 '
        'models. '
        'Needed by the calculate_net_match_threshold.py script.'
    )

    parser.add_argument('--gpus', default=range(8), nargs='+')

    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--net', nargs='+',
                        # default=['resnetv4_pytorch'],
                        required=True,
                        dest='NET')
    parser.add_argument('--ijbc-path',
                        dest='ijbc_path',
                        # default=os.environ['IJBC_PATH'] if 'IJBC_PATH' in
                        #     os.environ else None,
                        required=True,
                        help='Path to IJB-C directory. Should contain '
                        '"protocols/ijbc_metadata.csv" and referenced images.')
    parser.add_argument('--num-subj',
                        default=100,
                        dest='NUM_SUBJECTS',
                        help='Number of mated pairs. There is a mated pair in '
                        'each batch, hence corresponds to number of batches '
                        'for each random seed.',
                       )
    parser.add_argument('--seed', nargs='+',
                        dest='RANDOM_SEED',
                        default=range(10),
                       )

    parser.add_argument(
        '--script', dest='file',
        default=os.path.join(
            xfr.__path__[0],
            'inpainting_game',
            'net_mate_nonmate_dists.py',
        ))

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='force recalculation of distances of subject pairs'
    )

    args = parser.parse_args()
    run_experiments(vars(args))