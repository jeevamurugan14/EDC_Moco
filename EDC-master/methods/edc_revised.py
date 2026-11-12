import contextlib
import os
from collections import Counter
from copy import deepcopy
import pickle
from unittest import result
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import *
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.cuda.amp import autocast, GradScaler
from train_utils import Bn_Controller
from torch.autograd import Function
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import time

USE_CUDA = torch.cuda.is_available()
USE_MPS = False


class TimerEvent:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def record(self):
        self.start_time = time.perf_counter()

    def end(self):
        self.end_time = time.perf_counter()

    def elapsed_time(self, other_event):
        return (other_event.end_time - self.start_time) * 1000


class EDC:
    def __init__(self, model, it=0, num_eval_iter=1000, amap_reduction='max', tb_log=None, logger=None):
        super(EDC, self).__init__()
        self.loader = {}
        self.model = model
        if USE_MPS:
            self.model = self.model.float()
        self.num_eval_iter = num_eval_iter
        self.tb_log = tb_log
        self.optimizer = None
        self.scheduler = None
        self.diffusion_optimizer = None
        self.it = 0
        self.logger = logger
        self.print_fn = print if logger is None else logger.info
        self.amap_reduction = amap_reduction
        self.bn_controller = Bn_Controller()
        self.diffusion = None

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')

    def set_dset(self, dset):
        self.ulb_dset = dset

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, args, device, logger=None):
        self.model.train()
        if USE_MPS:
            start_batch = TimerEvent()
            end_batch = TimerEvent()
            start_run = TimerEvent()
            end_run = TimerEvent()
        else:
            start_batch = torch.cuda.Event(enable_timing=True)
            end_batch = torch.cuda.Event(enable_timing=True)
            start_run = torch.cuda.Event(enable_timing=True)
            end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record()
        best_eval_auc, best_it = 0.0, 0
        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        if args.resume == True:
            eval_dict = self.evaluate(args=args, device=device)
            print(eval_dict)

        train_log = []

        for idx, x, _, y, filename in tqdm(self.loader_dict['train']):
            if self.it > args.num_train_iter:
                break
            if USE_MPS:
                end_batch.end()
            else:
                end_batch.record()
                torch.cuda.synchronize()

            if USE_MPS:
                start_run.record()
            else:
                start_run.record()

            x = x.to(device)
            y = y.to(device, dtype=torch.float32)
            if USE_MPS:
                x = x.float()
                y = y.float()

            if args.amp and not USE_MPS:
                amp_cm = autocast
            else:
                amp_cm = contextlib.nullcontext

            if hasattr(self.model, 'set_iter'):
                self.model.set_iter(self.it)

            with amp_cm():
                result = self.model(x)
                edc_loss = result['loss'].mean()
                e3 = self.model.edc_encoder(x)[2]
                e3_norm = torch.nn.functional.normalize(e3, dim=1)
                if self.diffusion is not None:
                    diff_loss = self.diffusion.compute_loss(e3_norm.detach())
                    total_loss = edc_loss + args.lambda_diff * diff_loss
                else:
                    diff_loss = torch.tensor(0.0, device=device)
                    total_loss = edc_loss

            if args.amp:
                scaler.scale(total_loss).backward()
                if args.clip > 0:
                    scaler.unscale_(self.optimizer)
                    total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.zero_grad()
                edc_loss.backward()
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                self.optimizer.step()
                if self.diffusion is not None:
                    self.diffusion_optimizer.zero_grad()
                    diff_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), 1.0)
                    self.diffusion_optimizer.step()
                    self.diffusion.update_ema()
                self.scheduler.step()

            if USE_MPS:
                end_run.end()
            else:
                end_run.record()
                torch.cuda.synchronize()

            tb_dict = {}
            tb_dict['train/total_loss'] = total_loss.detach().item()
            tb_dict['train/edc_loss'] = edc_loss.detach().item()
            tb_dict['train/diff_loss'] = (diff_loss.detach().item() if self.diffusion is not None else 0.0)
            tb_dict['train/e1_std'] = result['e1_std'].detach().item()
            tb_dict['train/e2_std'] = result['e2_std'].detach().item()
            tb_dict['train/e3_std'] = result['e3_std'].detach().item()
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']

            if USE_MPS:
                tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
                tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.
            else:
                tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
                tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

            if (self.it + 1) % self.num_eval_iter == 0:
                eval_dict = self.evaluate(args=args, device=device)
                eval_dict_tb = eval_dict.copy()
                eval_dict_tb.pop('eval/y_true', None)
                eval_dict_tb.pop('eval/y_score', None)
                tb_dict.update(eval_dict_tb)
                save_path = os.path.join(args.save_dir, args.save_name)
                if tb_dict['eval/AUC'] > best_eval_auc:
                    best_eval_auc = tb_dict['eval/AUC']
                    best_it = self.it
                self.print_fn(
                    f"{self.it} iteration, {tb_dict}, BEST_EVAL_AUC: {best_eval_auc}, at {best_it} iters")
                if self.tb_log is not None:
                    self.tb_log.update(tb_dict, self.it)
                    tb_dict['it'] = self.it
                    train_log.append(tb_dict)

            self.it += 1
            del tb_dict
            start_batch.record()

        f_save = open(os.path.join(save_path, 'train_log.pkl'), 'wb')
        pickle.dump(train_log, f_save)
        f_save.close()

        eval_dict = self.evaluate(args=args, device=device, save_visual=False)
        eval_dict.update({'eval/best_auc': best_eval_auc, 'eval/best_it': best_it})
        return eval_dict

    def _reduce_amap(self, p_all):
        """
        Single shared reduction helper so the eval-loop scores and the
        val-loader threshold scores are ALWAYS computed the same way.
        (Previously the val-threshold path hardcoded .max(1), which silently
        broke amap_reduction='mean' (A4): eval scores were mean-pooled while
        the threshold came from max-pooled val scores, on a different scale,
        so every test image fell below threshold and was predicted NORMAL.)
        """
        flat = p_all.flatten(1)
        if self.amap_reduction == 'mean':
            return flat.mean(1)
        elif isinstance(self.amap_reduction, float):
            k = int(flat.shape[1] * self.amap_reduction)
            return torch.sort(flat, dim=1, descending=True)[0][:, :k].mean(dim=1)
        else:
            return flat.max(1)[0]

    @torch.no_grad()
    def evaluate(self, device, eval_loader=None, args=None, save_visual=False, val_loader=None):
        self.model.eval()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        if val_loader is None:
            val_loader = self.loader_dict.get('val', None)
        total_num = 0.0
        total_loss = 0.0
        y_true = []
        names = []
        y_prob = []
        y1_prob = []
        y2_prob = []
        y3_prob = []

        ema_cm = self.diffusion.ema_scope() if self.diffusion is not None else contextlib.nullcontext()

        with ema_cm:
            for _, x, xo, y, file_names in eval_loader:
                x, y = x.to(device), y.to(device, dtype=torch.float32)
                if USE_MPS:
                    x = x.float()
                    y = y.float()
                num_batch = x.shape[0]
                total_num += num_batch
                result = self.model(x)
                if self.amap_reduction == 'mean':
                    p_img = result['p_all'].flatten(1).mean(1)
                    p1_img = result['p1'].flatten(1).mean(1)
                    p2_img = result['p2'].flatten(1).mean(1)
                    p3_img = result['p3'].flatten(1).mean(1)
                elif isinstance(self.amap_reduction, float):
                    anomaly_map = result['p_all'].flatten(1)
                    p_img = torch.sort(anomaly_map, dim=1, descending=True)[0][:,
                            :int(anomaly_map.shape[1] * self.amap_reduction)].mean(dim=1)
                    anomaly_map = result['p1'].flatten(1)
                    p1_img = torch.sort(anomaly_map, dim=1, descending=True)[0][:,
                            :int(anomaly_map.shape[1] * self.amap_reduction)].mean(dim=1)
                    anomaly_map = result['p2'].flatten(1)
                    p2_img = torch.sort(anomaly_map, dim=1, descending=True)[0][:,
                            :int(anomaly_map.shape[1] * self.amap_reduction)].mean(dim=1)
                    anomaly_map = result['p3'].flatten(1)
                    p3_img = torch.sort(anomaly_map, dim=1, descending=True)[0][:,
                            :int(anomaly_map.shape[1] * self.amap_reduction)].mean(dim=1)
                else:
                    p_img = result['p_all'].flatten(1).max(1)[0]
                    p1_img = result['p1'].flatten(1).max(1)[0]
                    p2_img = result['p2'].flatten(1).max(1)[0]
                    p3_img = result['p3'].flatten(1).max(1)[0]

                y_true.extend(y.cpu().tolist())
                names.extend([str(n) for n in file_names])
                y_prob.extend(p_img.cpu().tolist())
                y1_prob.extend(p1_img.cpu().tolist())
                y2_prob.extend(p2_img.cpu().tolist())
                y3_prob.extend(p3_img.cpu().tolist())

                total_loss += result['loss'].detach().item() * num_batch

                if save_visual:
                    save_path = os.path.join(args.save_dir, args.save_name, 'heatmap')
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    anomaly_maps = F.interpolate(result['p_all'], size=xo.shape[1:3], mode='bilinear')
                    for i in range(xo.shape[0]):
                        image = xo[i].numpy().astype('uint8')
                        anomaly_map = anomaly_maps[i].cpu().permute(1, 2, 0).numpy()
                        file_name = file_names[i]
                        self.save_anomaly_map(anomaly_map, image, save_path, file_name)

        y_true  = np.array(y_true)
        y_prob  = np.array(y_prob)
        y1_prob = np.array(y1_prob)
        y2_prob = np.array(y2_prob)
        y3_prob = np.array(y3_prob)

        if val_loader is not None:
            val_scores = []
            for _, x, _, _, _ in val_loader:
                x = x.to(device)
                res = self.model(x)
                # FIX: use the SAME reduction as the eval-loop scores above,
                # via the shared _reduce_amap() helper, so the threshold and
                # the scores it's applied to are always on the same scale
                # (see _reduce_amap docstring for the bug this fixes).
                p_img = self._reduce_amap(res['p_all'])
                val_scores.extend(p_img.cpu().tolist())
            val_scores = np.array(val_scores)
            thresh = return_val_thr(val_scores, percentile=95.0)
            print(f"[Threshold] From val-normal 95th-pct ({len(val_scores)} images): {thresh:.4f}")
        else:
            normal_scores = y_prob[y_true == 0]
            thresh = float(np.percentile(normal_scores, 95.0))
            print(f"[Threshold] Fallback 95th-pct of test normals: {thresh:.4f}")

        y_pred = (y_prob >= thresh).astype(int)

        # ---- per-subset metrics: same model, same threshold ----
        names_l = [os.path.basename(n).lower() for n in names]
        for tag in ("benign", "malignant"):
            keep = np.array([(t == 0) or (tag in n) for t, n in zip(y_true, names_l)])
            if keep.sum() > (y_true == 0).sum():
                yt, ys, yp = y_true[keep], y_prob[keep], y_pred[keep]
                m = dict(
                    AUC=roc_auc_score(yt, ys), AP=average_precision_score(yt, ys),
                    acc=accuracy_score(yt, yp),
                    precision=float(((yp == 1) & (yt == 1)).sum() / ((yp == 1).sum() + 1e-8)),
                    recall=recall_score(yt, yp, zero_division=0),
                    f1=f1_score(yt, yp, zero_division=0),
                    specificity=specificity_score(yt, yp))
                print(f"[SUBSET {tag}] n_abn={int((yt == 1).sum())}  " +
                      "  ".join(f"{k}={v:.4f}" for k, v in m.items()))

        acc         = accuracy_score(y_true, y_pred)
        f1          = f1_score(y_true, y_pred, zero_division=0)
        recall      = recall_score(y_true, y_pred, zero_division=0)
        specificity = specificity_score(y_true, y_pred)
        precision   = float(np.sum((y_pred==1)&(y_true==1)) / (np.sum(y_pred==1)+1e-8))

        AUC  = roc_auc_score(y_true, y_prob)
        AUC1 = roc_auc_score(y_true, y1_prob)
        AUC2 = roc_auc_score(y_true, y2_prob)
        AUC3 = roc_auc_score(y_true, y3_prob)
        AP   = average_precision_score(y_true, y_prob)

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n===== CONFUSION MATRIX =====")
        print(f"TN={cm[0,0]}  FP={cm[0,1]}")
        print(f"FN={cm[1,0]}  TP={cm[1,1]}")
        print(f"Accuracy={acc:.4f}  Precision={precision:.4f}  Recall={recall:.4f}")
        print(f"F1={f1:.4f}  Specificity={specificity:.4f}")
        print(f"AUROC={AUC:.4f}  AP={AP:.4f}\n")

        self.model.train()
        return {'eval/loss': total_loss / total_num, 'eval/f1': f1, 'eval/recall': recall,
                'eval/specificity': specificity, 'eval/acc': acc,
                'eval/precision': precision,
                'eval/AUC': AUC, 'eval/AUC1': AUC1, 'eval/AUC2': AUC2, 'eval/AUC3': AUC3,
                'eval/AP': AP,
                'eval/best_thr': thresh, 'eval/y_true': y_true, 'eval/y_score': y_prob
                }

    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        self.model.train()
        torch.save({'model': self.model.state_dict()}, save_filename)
        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.print_fn('model loaded')

    def save_anomaly_map(self, anomaly_map, image, save_path, file_name):
        anomaly_map_norm = min_max_norm(anomaly_map)
        heatmap = cvt2heatmap(anomaly_map_norm * 255)
        hm_on_img = heatmap_on_image(heatmap, image)
        cv2.imwrite(os.path.join(save_path, file_name), hm_on_img)


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def heatmap_on_image(heatmap, image):
    out = np.float32(heatmap) / 255 + np.float32(image) / 255
    out = out / np.max(out)
    return np.uint8(255 * out)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)
    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr

def return_val_thr(val_normal_scores, percentile=95.0):
    """Leakage-free threshold from validation normal scores only."""
    return float(np.percentile(val_normal_scores, percentile))


def specificity_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred).astype(int)
    TN = ((y_pred == 0) & (y_true == 0)).sum()
    N = (y_true == 0).sum()
    return TN / (N + 1e-8)


if __name__ == "__main__":
    pass