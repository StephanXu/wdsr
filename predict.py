"""Trainer
"""
import argparse
import importlib
import logging
import os

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from apex import amp

import common.meters
import common.modes

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dataset',
      help='Dataset name.',
      default='div2k',
      type=str,
      required=False,
  )
  parser.add_argument(
      '--model',
      help='Model name.',
      default=None,
      type=str,
      required=True,
  )
  parser.add_argument(
      '--ckpt',
      help='File path to load checkpoint.',
      default=None,
      type=str,
  )
  parser.add_argument(
      '-v',
      '--verbose',
      action='count',
      default=0,
      help='Increasing output verbosity.',
  )
  parser.add_argument(
      '--outdir',
      help='Directory to store predict result.',
      default='./output',
      type=str,
  )
  parser.add_argument('--local_rank', default=0, type=int)
  parser.add_argument('--node_rank', default=0, type=int)

  # Parse arguments
  args, _ = parser.parse_known_args()
  logging.basicConfig(
      level=[logging.WARNING, logging.INFO, logging.DEBUG][args.verbose],
      format='%(asctime)s:%(levelname)s:%(message)s')
  dataset_module = importlib.import_module('datasets.' + args.dataset if args.dataset else 'datasets')
  dataset_module.update_argparser(parser)
  model_module = importlib.import_module('models.' + args.model if args.model else 'models')
  model_module.update_argparser(parser)
  params = parser.parse_args()
  logging.critical(params)

  torch.backends.cudnn.benchmark = True

  params.master_proc = True

  predict_dataset = dataset_module.get_dataset(common.modes.PREDICT, params)
  precit_data_loader = DataLoader(
    dataset=predict_dataset,
    num_workers=params.num_data_threads,
    batch_size=params.eval_batch_size,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    sampler=None,
  )
  model, criterion, optimizer, lr_scheduler, metrics = model_module.get_model_spec(params)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  criterion = criterion.to(device)

  params.opt_level = 'O0' # [experimental]
  model, optimizer = amp.initialize(model, optimizer, opt_level=params.opt_level, verbosity=params.verbose)

  if params.ckpt or os.path.exists(os.path.join(params.job_dir, 'latest.pth')):
    checkpoint = torch.load(
        params.ckpt or os.path.join(params.job_dir, 'latest.pth'),
        map_location=lambda storage, loc: storage.cuda())
    try:
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    except RuntimeError as e:
      logging.critical(e)
    latest_epoch = checkpoint['epoch']
  else:
    latest_epoch = 0

  transform = torchvision.transforms.ToPILImage()

  with torch.no_grad():
      model.eval()
      for data, filename in precit_data_loader:
        data = data.to(device, non_blocking=True)
        output = model(data)
        transform(output[0]).save(os.path.join(params.outdir, filename[0]))
