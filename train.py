from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.kitti_yolo_dataset import KittiYOLODataset
from eval_mAP import evaluate

from terminaltables import AsciiTable
import os, sys, time, datetime, argparse

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--train_data_batch_size", type=int, default=10, help="size of each image batch")
    parser.add_argument("--test_data_batch_size",type=int, default=10,help="size of each image batch for test loader")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/complex_yolov3.cfg", help="path to model definition file")
    parser.add_argument("--pretrained_weights", type=str,default=None, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=cnf.BEV_WIDTH, help="size of each image dimension")
    parser.add_argument("--evaluation_interval", type=int, default=10, help="interval evaluations on validation set")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)
    class_names = load_classes("data/classes.names")

    # Initiate model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = KittiYOLODataset(
        cnf.root_dir,
        split='train',
        mode='TRAIN',
        folder='training',
        data_aug=True,
        multiscale=opt.multiscale_training
    )

    dataloader = DataLoader(
        dataset,
        opt.train_data_batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "im",
        "re",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    train_loss=float("inf")
    for epoch in range(0, opt.epochs, 1):
        #epoch从0开始
        model.train()
        start_time = time.time()

        #returns BEV figure and bboxes
        for batch_i, (_, imgs, targets) in enumerate(tqdm(dataloader,desc="Training")):

            #考虑显存过小的时候进行梯度累计,我这里将会使用它进行Loss的计算
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            train_loss_cur_epoch=0.0
            optimizer.zero_grad()
            loss, outputs = model(imgs, targets)
            loss.backward()
            optimizer.step()

            train_loss_cur_epoch+=loss.item()

            # ----------------
            #   Log and save progress after an epoch finished
            # ----------------
            if int((batches_done+1)/len(dataloader)) == (epoch+1):

                train_loss_cur_epoch=train_loss_cur_epoch/len(dataloader)

                log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch+1, opt.epochs, batch_i+1, len(dataloader))

                metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

                # Log metrics at each YOLO layer
                for i, metric in enumerate(metrics):
                    formats = {m: "%.6f" for m in metrics}
                    formats["grid_size"] = "%2d"
                    formats["cls_acc"] = "%.2f%%"
                    row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                    metric_table += [[metric, *row_metrics]]

                    # Tensorboard logging
                    tensorboard_log = []
                    for j, yolo in enumerate(model.yolo_layers):
                        for name, metric in yolo.metrics.items():
                            if name != "grid_size":
                                tensorboard_log += [(f"{name}_{j+1}", metric)]
                    tensorboard_log += [("loss", loss.item())]
                    logger.list_of_scalars_summary(tensorboard_log, batches_done)

                log_str += AsciiTable(metric_table).table
                log_str += f"\nTotal loss {loss.item()}"

                # Determine approximate time left for epoch
                epoch_batches_left = len(dataloader) - (batch_i + 1)
                time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
                log_str += f"\n---- ETA {time_left}"
                print(log_str)
                print(f"epoch_loss:{train_loss_cur_epoch}")
                #save best.pth    
                if train_loss_cur_epoch<train_loss:
                    torch.save(model.state_dict(), f"checkpoints/best.pth" )
                else:
                    pass
                if epoch==opt.epochs-1:
                    torch.save(model.state_dict(),f"checkpoints/last.pth")

        #每几轮检查一次测试集的mAP，防止过拟合，我会根据时间确定
        if (epoch+1) % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model on Testing Data----")
            # Evaluate the model on the validation set
            optimizer.zero_grad()
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=opt.test_data_batch_size,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")
