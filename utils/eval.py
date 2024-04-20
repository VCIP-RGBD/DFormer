import pprint
import time
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
from utils.dataloader.dataloader import get_val_loader
from models.builder import EncoderDecoder as segmodel
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.engine.engine import Engine
from utils.engine.logger import get_logger
from tensorboardX import SummaryWriter
from val_mm import evaluate, evaluate_msf
from importlib import import_module

# from eval import evaluate_mid

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="train config file path")
parser.add_argument("--gpus", help="used gpu number")
# parser.add_argument('-d', '--devices', default='0,1', type=str)
parser.add_argument("-v", "--verbose", default=False, action="store_true")
parser.add_argument("--epochs", default=0)
parser.add_argument("--show_image", "-s", default=False, action="store_true")
parser.add_argument("--save_path", default=None)
parser.add_argument("--checkpoint_dir")
parser.add_argument("--continue_fpath")
parser.add_argument("--sliding", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--compile", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--compile_mode", default="default")
parser.add_argument("--syncbn", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--mst", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
# parser.add_argument('--save_path', '-p', default=None)

# os.environ['MASTER_PORT'] = '169710'
torch.set_float32_matmul_precision("high")
import torch._dynamo

torch._dynamo.config.suppress_errors = True
# torch._dynamo.config.automatic_dynamic_shapes = False

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    config = getattr(import_module(args.config), "C")
    # exec("from " + args.config + " import C as config")
    logger = get_logger(config.log_dir, config.log_file, rank=engine.local_rank)

    cudnn.benchmark = True

    if args.mst:
        val_loader, val_sampler = get_val_loader(
            engine, RGBXDataset, config, val_batch_size=int(config.batch_size * 1.5)
        )
    else:
        val_loader, val_sampler = get_val_loader(
            engine, RGBXDataset, config, val_batch_size=int(config.batch_size * 2)
        )
    logger.info(f"val dataset len:{len(val_loader)*int(args.gpus)}")

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + "/{}".format(
            time.strftime("%b%d_%d-%H-%M", time.localtime())
        )
        generate_tb_dir = config.tb_dir + "/tb"
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)
        pp = pprint.PrettyPrinter(indent=4)
        logger.info("config: \n" + pp.pformat(config))

    criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=config.background)

    if args.syncbn:
        BatchNorm2d = nn.SyncBatchNorm
        logger.info("using syncbn")
    else:
        BatchNorm2d = nn.BatchNorm2d
        logger.info("using regular bn")

    model = segmodel(
        cfg=config,
        criterion=criterion,
        norm_layer=BatchNorm2d,
        syncbn=args.syncbn,
    )

    weight = torch.load(args.continue_fpath)["model"]

    logger.info(f"load model from {args.continue_fpath}")
    model.load_state_dict(weight,strict=False)

    if engine.distributed:
        logger.info(".............distributed training.............")
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(
                model,
                device_ids=[engine.local_rank],
                output_device=engine.local_rank,
                find_unused_parameters=True,
            )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    if args.compile:
        model = torch.compile(model, backend="inductor", mode=args.compile_mode)

    torch.cuda.empty_cache()
    if args.amp:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            if engine.distributed:
                with torch.no_grad():
                    model.eval()
                    device = torch.device("cuda")
                    if args.mst:
                        all_metrics = evaluate_msf(
                            model,
                            val_loader,
                            config,
                            device,
                            [0.5, 0.75, 1.0, 1.25, 1.5],
                            True,
                            engine,
                            sliding=args.sliding,
                        )
                    else:
                        all_metrics = evaluate(
                            model,
                            val_loader,
                            config,
                            device,
                            engine,
                            sliding=args.sliding,
                        )
                    if engine.local_rank == 0:
                        metric = all_metrics[0]
                        for other_metric in all_metrics[1:]:
                            metric.update_hist(other_metric.hist)
                        ious, miou = metric.compute_iou()
                        acc, macc = metric.compute_pixel_acc()
                        f1, mf1 = metric.compute_f1()
                        logger.info(f"miou:{miou}, macc:{macc}, mf1:{mf1}")
                        logger.info(f"ious:{ious}")
            elif not engine.distributed:
                with torch.no_grad():
                    model.eval()
                    device = torch.device("cuda")
                    if args.mst:
                        metric = evaluate_msf(
                            model,
                            val_loader,
                            config,
                            device,
                            [0.5, 0.75, 1.0, 1.25, 1.5],
                            True,
                            engine,
                            sliding=args.sliding,
                        )
                    else:
                        metric = evaluate(
                            model,
                            val_loader,
                            config,
                            device,
                            engine,
                            sliding=args.sliding,
                        )
                    ious, miou = metric.compute_iou()
                    acc, macc = metric.compute_pixel_acc()
                    f1, mf1 = metric.compute_f1()
                    logger.info(f"miou:{miou}, macc:{macc}, mf1:{mf1}")
                    logger.info(f"ious:{ious}")
    else:
        if engine.distributed:
            with torch.no_grad():
                model.eval()
                device = torch.device("cuda")
                if args.mst:
                    all_metrics = evaluate_msf(
                        model,
                        val_loader,
                        config,
                        device,
                        [0.5, 0.75, 1.0, 1.25, 1.5],
                        True,
                        engine,
                        sliding=args.sliding,
                    )
                else:
                    all_metrics = evaluate(
                        model,
                        val_loader,
                        config,
                        device,
                        engine,
                        sliding=args.sliding,
                    )
                if engine.local_rank == 0:
                    metric = all_metrics[0]
                    for other_metric in all_metrics[1:]:
                        metric.update_hist(other_metric.hist)
                    ious, miou = metric.compute_iou()
                    acc, macc = metric.compute_pixel_acc()
                    f1, mf1 = metric.compute_f1()
                    logger.info(f"miou:{miou}, macc:{macc}, mf1:{mf1}")
                    logger.info(f"ious:{ious}")
        elif not engine.distributed:
            with torch.no_grad():
                model.eval()
                device = torch.device("cuda")
                if args.mst:
                    metric = evaluate_msf(
                        model,
                        val_loader,
                        config,
                        device,
                        [0.5, 0.75, 1.0, 1.25, 1.5],
                        True,
                        engine,
                        sliding=args.sliding,
                    )
                else:
                    metric = evaluate(
                        model,
                        val_loader,
                        config,
                        device,
                        engine,
                        sliding=args.sliding,
                    )
                ious, miou = metric.compute_iou()
                acc, macc = metric.compute_pixel_acc()
                f1, mf1 = metric.compute_f1()
                logger.info(f"miou:{miou}, macc:{macc}, mf1:{mf1}")
                logger.info(f"ious:{ious}")
    logger.info("end testing")
