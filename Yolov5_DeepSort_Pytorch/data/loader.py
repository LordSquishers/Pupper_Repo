from torch.utils.data import DataLoader

from data.trajectories import TrajectoryDataset, seq_collate


def data_loader(args, path):
    dset = TrajectoryDataset(
        path,
        obs_len=8,  # args.obs_len,
        pred_len=8,  # args.pred_len,
        skip=1,  # args.skip,
        delim='\t')  # args.delim)

    loader = DataLoader(
        dset,
        batch_size=64,  # args.batch_size,
        shuffle=False,
        num_workers=4,  # args.loader_num_workers,
        collate_fn=seq_collate,
        pin_memory=True)
    return dset, loader
