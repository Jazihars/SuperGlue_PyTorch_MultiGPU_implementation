from pathlib import Path
import argparse
import time
import random
import numpy as np
import matplotlib.cm as cm
import torch
import torch.cuda
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd import Variable
from torch.utils.data import DataLoader
from load_data import SparseDataset
import os
import torch.multiprocessing
from tqdm import tqdm
from torch.utils.data.distributed import (
    DistributedSampler,
)  # 负责分布式dataloader创建，也就是实现上面提到的partition。

from models.utils import (
    compute_pose_error,
    compute_epipolar_error,
    estimate_pose,
    make_matching_plot,
    error_colormap,
    AverageTimer,
    pose_auc,
    read_image,
    rotate_intrinsics,
    rotate_pose_inplane,
    scale_intrinsics,
    read_image_modified,
)

from models.superpoint import SuperPoint
from models.superglue import SuperGlue
from models.matchingForTraining import MatchingForTraining

torch.set_grad_enabled(True)
torch.multiprocessing.set_sharing_strategy("file_system")

parser = argparse.ArgumentParser(
    description="Image pair matching and pose evaluation with SuperGlue",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--viz", action="store_true", help="Visualize the matches and dump the plots"
)
parser.add_argument(
    "--eval",
    action="store_true",
    help="Perform the evaluation" " (requires ground truth pose and intrinsics)",
)

parser.add_argument(
    "--superglue",
    choices={"indoor", "outdoor"},
    default="indoor",
    help="SuperGlue weights",
)
parser.add_argument(
    "--max_keypoints",
    type=int,
    default=1024,
    help="Maximum number of keypoints detected by Superpoint"
    " ('-1' keeps all keypoints)",
)
parser.add_argument(
    "--keypoint_threshold",
    type=float,
    default=0.005,
    help="SuperPoint keypoint detector confidence threshold",
)
parser.add_argument(
    "--nms_radius",
    type=int,
    default=4,
    help="SuperPoint Non Maximum Suppression (NMS) radius" " (Must be positive)",
)
parser.add_argument(
    "--sinkhorn_iterations",
    type=int,
    default=20,
    help="Number of Sinkhorn iterations performed by SuperGlue",
)
parser.add_argument(
    "--match_threshold", type=float, default=0.2, help="SuperGlue match threshold"
)

parser.add_argument(
    "--resize",
    type=int,
    nargs="+",
    default=[640, 480],
    help="Resize the input image before running inference. If two numbers, "
    "resize to the exact dimensions, if one number, resize the max "
    "dimension, if -1, do not resize",
)
parser.add_argument(
    "--resize_float",
    action="store_true",
    help="Resize the image after casting uint8 to float",
)

parser.add_argument(
    "--cache",
    action="store_true",
    help="Skip the pair if output .npz files are already found",
)
parser.add_argument(
    "--show_keypoints",
    action="store_true",
    help="Plot the keypoints in addition to the matches",
)
parser.add_argument(
    "--fast_viz",
    action="store_true",
    help="Use faster image visualization based on OpenCV instead of Matplotlib",
)
parser.add_argument(
    "--viz_extension",
    type=str,
    default="png",
    choices=["png", "pdf"],
    help="Visualization file extension. Use pdf for highest-quality.",
)

parser.add_argument(
    "--opencv_display",
    action="store_true",
    help="Visualize via OpenCV before saving output images",
)
parser.add_argument(
    "--eval_pairs_list",
    type=str,
    default="assets/scannet_sample_pairs_with_gt.txt",
    help="Path to the list of image pairs for evaluation",
)
parser.add_argument(
    "--shuffle", action="store_true", help="Shuffle ordering of pairs before processing"
)
parser.add_argument(
    "--max_length", type=int, default=-1, help="Maximum number of pairs to evaluate"
)

parser.add_argument(
    "--eval_input_dir",
    type=str,
    default="assets/scannet_sample_images/",
    help="Path to the directory that contains the images",
)
parser.add_argument(
    "--eval_output_dir",
    type=str,
    default="dump_match_pairs/",
    help="Path to the directory in which the .npz results and optional,"
    "visualizations are written",
)
parser.add_argument("--learning_rate", type=int, default=0.0001, help="Learning rate")

parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
parser.add_argument(
    "--train_path",
    type=str,
    default="/data/XXX/coco2014/train2014/",
    help="Path to the directory of training imgs.",
)
parser.add_argument("--epoch", type=int, default=1, help="Number of epoches")
parser.add_argument(
    "--local_rank",
    type=int,
    default=-1,
    help="local rank for DistributedDataParallel",
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    opt = parser.parse_args()

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    # 每个进程根据自己的local_rank设置应该使用的GPU
    torch.cuda.set_device(opt.local_rank)

    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )
    dist.barrier()

    seed = 0 + dist.get_rank()
    set_seed(seed)

    # make sure the flags are properly used
    assert not (
        opt.opencv_display and not opt.viz
    ), "Must use --viz with --opencv_display"
    assert not (
        opt.opencv_display and not opt.fast_viz
    ), "Cannot use --opencv_display without --fast_viz"
    assert not (opt.fast_viz and not opt.viz), "Must use --viz with --fast_viz"
    assert not (
        opt.fast_viz and opt.viz_extension == "pdf"
    ), "Cannot use pdf extension with --fast_viz"

    # store viz results
    eval_output_dir = Path(opt.eval_output_dir)
    eval_output_dir.mkdir(exist_ok=True, parents=True)
    if dist.get_rank() == 0:
        print(
            "Will write visualization images to",
            'directory "{}"'.format(eval_output_dir),
        )

    config = {
        "superpoint": {
            "nms_radius": opt.nms_radius,
            "keypoint_threshold": opt.keypoint_threshold,
            "max_keypoints": opt.max_keypoints,
        },
        "superglue": {
            "weights": opt.superglue,
            "sinkhorn_iterations": opt.sinkhorn_iterations,
            "match_threshold": opt.match_threshold,
        },
    }

    # load training data
    train_dataset = SparseDataset(opt.train_path, opt.max_keypoints)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=opt.batch_size,
        drop_last=True,
    )

    superglue = SuperGlue(config.get("superglue", {}))

    if torch.cuda.is_available():
        superglue.cuda()  # make sure it trains on GPU
    else:
        print("### CUDA not available ###")

    # superglue.to(torch.device("cuda:0"))
    superglue = DDP(
        superglue,
        device_ids=[opt.local_rank],
        find_unused_parameters=True,
    )

    optimizer = optim.Adam(superglue.parameters(), lr=opt.learning_rate)
    mean_loss = []

    # start training
    start_time = time.time()
    for epoch in range(1, opt.epoch + 1):
        train_loader.sampler.set_epoch(epoch)
        epoch_loss = 0
        superglue.double().train()
        for i, pred in enumerate(train_loader):
            for k in pred:
                if k != "file_name" and k != "image0" and k != "image1":
                    if type(pred[k]) == torch.Tensor:
                        pred[k] = pred[k].cuda()
                    else:
                        pred[k] = torch.stack(pred[k]).cuda()

            data = superglue(pred)
            for k, v in pred.items():
                pred[k] = v[0]
            pred = {**pred, **data}

            if pred["skip_train"] == True:  # image has no keypoint
                continue

            # process loss
            Loss = pred["loss"]
            epoch_loss += Loss.item()
            mean_loss.append(Loss)

            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

            torch.cuda.synchronize()

            # for every 50 images, print progress and visualize the matches
            if (i + 1) % 50 == 0:
                total_time = time.time() - start_time
                if dist.get_rank() == 0:
                    print(
                        "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                            epoch,
                            opt.epoch,
                            i + 1,
                            len(train_loader),
                            torch.mean(torch.stack(mean_loss)).item(),
                        )
                    )
                mean_loss = []

                ### eval ###
                # Visualize the matches.
                superglue.eval()
                image0, image1 = (
                    pred["image0"].cpu().numpy()[0] * 255.0,
                    pred["image1"].cpu().numpy()[0] * 255.0,
                )
                kpts0, kpts1 = (
                    pred["keypoints0"].cpu().numpy()[0],
                    pred["keypoints1"].cpu().numpy()[0],
                )
                matches, conf = (
                    pred["matches0"].cpu().detach().numpy(),
                    pred["matching_scores0"].cpu().detach().numpy(),
                )
                image0 = read_image_modified(image0, opt.resize, opt.resize_float)
                image1 = read_image_modified(image1, opt.resize, opt.resize_float)
                valid = matches > -1
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]
                mconf = conf[valid]
                viz_path = eval_output_dir / "{}_matches.{}".format(
                    str(i), opt.viz_extension
                )
                color = cm.jet(mconf)
                stem = pred["file_name"]
                text = []

                make_matching_plot(
                    image0,
                    image1,
                    kpts0,
                    kpts1,
                    mkpts0,
                    mkpts1,
                    color,
                    text,
                    viz_path,
                    stem,
                    stem,
                    opt.show_keypoints,
                    opt.fast_viz,
                    opt.opencv_display,
                    "Matches",
                )

            # process checkpoint for every 5e3 images
            if dist.get_rank() == 0 and (i + 1) % 5e3 == 0:
                model_out_path = "model_epoch_{}_checkpoint_{}.pth".format(epoch, i + 1)
                torch.save(superglue.module.state_dict(), model_out_path)
                print(
                    "Epoch [{}/{}], Step [{}/{}], Checkpoint saved to {}".format(
                        epoch, opt.epoch, i + 1, len(train_loader), model_out_path
                    )
                )

        # save checkpoint when an epoch finishes
        epoch_loss /= len(train_loader)
        model_out_path = "model_epoch_{}.pth".format(epoch)
        torch.save(superglue.module.state_dict(), model_out_path)
        print(
            "Epoch [{}/{}] done. Epoch Loss {}. Checkpoint saved to {}".format(
                epoch, opt.epoch, epoch_loss, model_out_path
            )
        )
    total_time = time.time() - start_time
    print(f"一个epoch的训练时间是：{total_time}")
