import datetime
import numpy as np
import os
import time
import pickle
import random
from pathlib import Path
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
import util.lr_decay as lrd
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.data_loader import HimopDataset
from model import models_surv
from engine_finetune import train_one_epoch, evaluate
from util.options import get_args_parser_finetune, logger

exptype = "luad"

def set_seed(seed):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    args.data_dir = args.data_dir.replace("xxx", args.cancer_type)
    data_cv_splits = pickle.load(open(args.data_dir, 'rb'))
    start_time = time.time()
    best_predict = []
    for _k, data in data_cv_splits.items():

        logger.info("**************************************************************")
        logger.info("************** 5-folds Cross-validation (%d/%d) ***************" % (_k+1, len(data_cv_splits.items())))
        logger.info("**************************************************************")

        # fix the seed for reproducibility
        set_seed(seed=args.seed)
        logger.info(f"set random seed: {args.seed}")
        logger.info(f"exptype: {exptype}")

        dataset_train = HimopDataset(
            data=data,
            split="train",
        )

        dataset_val = HimopDataset(
            data=data,
            split="validation",
        )

        dataset_test = HimopDataset(
            data=data,
            split="test",
        )
        print(dataset_train)

        global_rank = misc.get_rank()
        if global_rank == 0 and args.log_dir is not None and not args.eval:
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=args.log_dir)
        else:
            log_writer = None

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, #sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            shuffle=True
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, #sampler=sampler_val,
            batch_size=10,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            shuffle=False
        )

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, #sampler=sampler_test,
            batch_size=10,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            shuffle=False
        )

        mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            print("Mixup is activated!")
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.nb_classes)

        model = models_surv.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )

        # args.finetune = None
        if args.finetune and not args.eval:
            checkpoint = torch.load(args.finetune, map_location='cpu')

            logger.info("Load pre-trained checkpoint from: {}".format(args.finetune))
            checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # load pre-trained model
            model.load_state_dict(checkpoint_model, strict=False)

            # manually initialize fc layer
            trunc_normal_(model.head.weight, std=2e-5)

        model.to(device)

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # print("Model = %s" % str(model_without_ddp))
        logger.info('number of params (M): %.2f' % (n_parameters / 1.e6))

        if args.batch_size > len(dataset_train):
            args.batch_size = len(dataset_train)

        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

        if args.lr is None:  # only base_lr is specified
            # args.lr = args.blr * eff_batch_size / 256
            args.lr = args.blr

        logger.info("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        logger.info("actual lr: %.2e" % args.lr)
        logger.info("accumulate grad iterations: %d" % args.accum_iter)
        logger.info("effective batch size: %d" % eff_batch_size)

        # build optimizer with layer-wise lr decay (lrd)
        param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
            no_weight_decay_list=model_without_ddp.no_weight_decay(),
            layer_decay=args.layer_decay
        )
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
        loss_scaler = NativeScaler()

        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

        if args.eval:
            test_stats = evaluate(data_loader_val, model, device)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            exit(0)

        logger.info(f"Start training for {args.epochs} epochs")
        max_c_index_val = 0.0
        best_predict_split = []
        list_for_early_stop = []
        list_for_reload_best_model = []
        for epoch in range(args.start_epoch, args.epochs):
            if len(list_for_early_stop) >= 12:
                logger.info(f"Now is early stopping; step: {len(list_for_early_stop)}")
                break

            if len(list_for_reload_best_model) >= 3:
                list_for_reload_best_model = []
                logger.info("reload the best model")
                model.load_state_dict(torch.load(f'./output_finetune_dir/best_model_{exptype}.pth'))

            logger.info(f"Epoch: {epoch}")

            # train_one_epoch
            train_stats, model = train_one_epoch(
                model,
                data_loader_train,
                optimizer,
                device,
                epoch,
                loss_scaler,
                args.clip_grad, mixup_fn,
                log_writer=log_writer,
                args=args
            )

            test_stats, pred, censored, survival = evaluate(data_loader_test, model, device)
            val_stats = test_stats

            logger.info(f"- validation c-index: {val_stats['c-index']:.4f}, p-value: {val_stats['p-value']:.8f} -----------")
            logger.info(f"------- test c-index: {test_stats['c-index']:.4f}, p-value: {test_stats['p-value']:.8f} -----------")

            list_for_early_stop.append(val_stats['c-index'])
            list_for_reload_best_model.append(val_stats['c-index'])
            if val_stats['c-index'] > max_c_index_val:
                max_c_index_val = val_stats['c-index']

                list_for_early_stop = []
                list_for_reload_best_model = []

                best_predict_split = [round(train_stats['c-index'], 4),
                                      round(val_stats['c-index'], 4),
                                      round(test_stats['c-index'], 4),
                                      round(train_stats['p-value'], 10),
                                      round(val_stats['p-value'], 10),
                                      round(test_stats['p-value'], 10),]
                logger.info("************** hit *************")
                logger.info(f"--- val: max c-index: {max_c_index_val:.4f}, "
                            f"val: current c-index: {val_stats['c-index']:.4f}, "
                            f"test: c-index: {test_stats['c-index']:.4f}, "
                            f"p-value: {test_stats['p-value']:.10f} ---")

                # if args.output_dir:
                #     misc.save_model(
                #         args=args, model=model, model_without_ddp=model_without_ddp,
                #         optimizer=optimizer,
                #         loss_scaler=loss_scaler, epoch=epoch)

                if not os.path.exists(os.path.join(args.output_dir, f"predict_result_{args.cancer_type}_{exptype}")):
                    os.makedirs(os.path.join(args.output_dir, f"predict_result_{args.cancer_type}_{exptype}"))

                result_dir = os.path.join(args.output_dir, f"predict_result_{args.cancer_type}_{exptype}", f"predict_kfold_{_k}.csv")
                res = {
                    "predict": pred.flatten(0).cpu().detach().numpy(),
                    "censored": censored.cpu().detach().numpy(),
                    "survival": survival.cpu().detach().numpy()
                }
                pd.DataFrame(res).to_csv(result_dir, index=False)

                torch.save(model.state_dict(), f'./output_finetune_dir/best_model_{exptype}.pth')

            log_stats = {'epoch': epoch, 'kfold': _k,
                         **{f'train_{k}': round(v, 8) for k, v in train_stats.items()},
                         **{f'val_{k}': round(v, 8) for k, v in val_stats.items()},
                         **{f'test_{k}': round(v, 8) for k, v in test_stats.items()},
                         }
            logger.info(log_stats)

            if log_writer is not None:
                log_writer.add_scalar('perf/val_cindex', round(val_stats['c-index'],4), epoch)
                log_writer.add_scalar('perf/val_pvalue', round(val_stats['p-value'],4), epoch)
                log_writer.add_scalar('perf/test_cindex', round(test_stats['c-index'],4), epoch)
                log_writer.add_scalar('perf/test_pvalue', round(test_stats['p-value'],4), epoch)

            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()

            logger.info("")

        best_predict.append(best_predict_split)
        logger.info(f"best performance for kfold {_k}: {best_predict_split}")
        logger.info("\n\n")

    logger.info(f"best performance for all splits: {best_predict}")
    logger.info(f"train cindex: {np.array(best_predict)[:,0]}, mean: {np.mean(np.array(best_predict)[:,0])}")
    logger.info(f"val   cindex: {np.array(best_predict)[:, 1]}, mean: {np.mean(np.array(best_predict)[:, 1])}, std: {np.std(np.array(best_predict)[:, 1])}")
    logger.info(f"test  cindex: {np.array(best_predict)[:, 2]}, mean: {np.mean(np.array(best_predict)[:, 2])}, std: {np.std(np.array(best_predict)[:, 2])}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    df_all = pd.DataFrame()
    for i in range(5):
        result_dir = os.path.join(args.output_dir, f"predict_result_{args.cancer_type}_{exptype}", f"predict_kfold_{i}.csv")
        df = pd.read_csv(result_dir)
        df_all = pd.concat([df_all, df], axis=0)
    df_all.to_csv(os.path.join(args.output_dir, f"predict_result_{args.cancer_type}_{exptype}", f"predict_result.csv"), index=False)

if __name__ == '__main__':
    args = get_args_parser_finetune()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
