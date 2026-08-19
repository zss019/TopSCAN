import argparse
import os
import sys
import time
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from topscan.data.graph_data import load_data, set_seed
from topscan.utils.loss_function import (
    EarlyStopping,
    adjust_learning_rate,
    cross_entropy,
    get_metric,
)
from topscan.models.topscan_model import TopSCAN


def parse_str_list(raw):
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_int_list(raw):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def normalize_alpha_by_fusion(fusion, alpha):
    if str(fusion).lower() == "residual":
        return float(alpha)
    return None


def parse_backbone_specs(raw):
    specs = []
    if not str(raw).strip():
        return specs
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [x.strip() for x in item.split(":")]
        if len(parts) != 5:
            raise ValueError(
                "Invalid --grid_backbones format. Expect "
                "'gnn_type:n_layers:n_hidden:lr:dropout;...'"
            )
        gnn_type, n_layers, n_hidden, lr, dropout = parts
        specs.append((gnn_type, int(n_layers), int(n_hidden), float(lr), float(dropout)))
    return specs


def build_grid_configs(args):
    if str(args.grid_backbones).strip():
        backbones = parse_backbone_specs(args.grid_backbones)
    else:
        gnn_types = parse_str_list(args.grid_gnn_types)
        layers_list = parse_int_list(args.grid_layers)
        hidden_units = parse_int_list(args.grid_hidden_units)
        lrs = parse_float_list(args.grid_lrs)
        dropouts = parse_float_list(args.grid_dropouts)
        backbones = []
        for gnn_type in gnn_types:
            for n_layers in layers_list:
                for n_hidden in hidden_units:
                    for lr in lrs:
                        for dropout in dropouts:
                            backbones.append((gnn_type, n_layers, n_hidden, lr, dropout))
    fusions = parse_str_list(args.grid_fusions)
    residual_alphas = parse_float_list(args.grid_residual_alphas)
    cma_num_heads_list = parse_int_list(args.grid_cma_num_heads)
    cma_head_dims = parse_int_list(args.grid_cma_head_dims)
    cma_attn_dropouts = parse_float_list(args.grid_cma_attn_dropouts)
    cma_norm_bys = parse_str_list(args.grid_cma_norm_bys)
    configs = []
    import itertools

    for gnn_type, n_layers, n_hidden, lr, dropout in backbones:
        for cma_num_heads, cma_head_dim, cma_attn_dropout, cma_norm_by in itertools.product(
            cma_num_heads_list, cma_head_dims, cma_attn_dropouts, cma_norm_bys
        ):
            for fusion in fusions:
                fusion = str(fusion)
                alphas = residual_alphas if fusion == "residual" else [0.5]
                for alpha in alphas:
                    configs.append(
                        {
                            "gnn_type": gnn_type,
                            "n_layers": int(n_layers),
                            "n_hidden": int(n_hidden),
                            "lr": float(lr),
                            "dropout": float(dropout),
                            "fusion": fusion,
                            "warmup_alpha": normalize_alpha_by_fusion(fusion, alpha),
                            "cma_num_heads": int(cma_num_heads),
                            "cma_head_dim": int(cma_head_dim),
                            "cma_attn_dropout": float(cma_attn_dropout),
                            "cma_norm_by": str(cma_norm_by),
                        }
                    )
    return configs


def _resolve_checkpoint_dir(args):
    raw_dir = str(getattr(args, "checkpoint_dir", "")).strip()
    if raw_dir:
        checkpoint_dir = Path(raw_dir)
    else:
        output_file = str(getattr(args, "output_file", "topscan_results.csv")).strip() or "topscan_results.csv"
        output_path = Path(output_file)
        checkpoint_dir = output_path.parent / f"{output_path.stem}_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def _sanitize_float_token(value):
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _build_checkpoint_filename(args, run_idx):
    prefix = str(getattr(args, "checkpoint_prefix", "")).strip()
    if prefix:
        prefix = prefix.rstrip("_") + "_"
    data_name = str(getattr(args, "data_name", "dataset")).lower().replace(" ", "_")
    gnn_type = str(getattr(args, "gnn_type", "gnn")).lower()
    fusion = str(getattr(args, "warmup_fusion", "concat")).lower()
    parts = [
        f"{prefix}{data_name}",
        gnn_type,
        f"l{int(getattr(args, 'n_layers', 0))}",
        f"h{int(getattr(args, 'n_hidden', 0))}",
        f"lr{_sanitize_float_token(getattr(args, 'lr', 0.0))}",
        f"do{_sanitize_float_token(getattr(args, 'dropout', 0.0))}",
        f"fusion_{fusion}",
        f"run{int(run_idx)}",
    ]
    if hasattr(args, "cma_num_heads"):
        parts.append(f"heads{int(getattr(args, 'cma_num_heads', 0))}")
    if hasattr(args, "cma_head_dim"):
        parts.append(f"dim{int(getattr(args, 'cma_head_dim', 0))}")
    if hasattr(args, "policy_top_p"):
        parts.append(f"topp{_sanitize_float_token(getattr(args, 'policy_top_p', 0.0))}")
    if hasattr(args, "policy_mode"):
        parts.append(str(getattr(args, "policy_mode", "on")).lower())
    return "_".join(parts) + ".pth"


def _save_best_checkpoint(
    model, args, run_idx, epoch, best_metric, best_val_acc, best_val_f1, test_acc, test_f1
):
    checkpoint_dir = _resolve_checkpoint_dir(args)
    checkpoint_path = checkpoint_dir / _build_checkpoint_filename(args, run_idx)
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "best_val_acc": float(best_val_acc),
        "best_val_f1": float(best_val_f1),
        "test_acc_at_best": float(test_acc),
        "test_f1_at_best": float(test_f1),
        "args": vars(args).copy(),
    }
    torch.save(payload, checkpoint_path)
    return str(checkpoint_path.resolve())


def train(model, graph, cross_graph, text_feat, image_feat, labels, train_idx, optimizer, label_smoothing):
    model.train()
    optimizer.zero_grad()
    pred = model(graph, cross_graph, text_feat, image_feat)
    loss = cross_entropy(pred[train_idx], labels[train_idx], label_smoothing=label_smoothing)
    loss.backward()
    optimizer.step()
    return loss.item(), pred


@torch.no_grad()
def evaluate(model, graph, cross_graph, text_feat, image_feat, labels, train_idx, val_idx, test_idx,
             label_smoothing=0.1, average="macro"):
    model.eval()
    pred = model(graph, cross_graph, text_feat, image_feat)
    val_loss = cross_entropy(pred[val_idx], labels[val_idx], label_smoothing).item()
    test_loss = cross_entropy(pred[test_idx], labels[test_idx], label_smoothing).item()
    pred_train = torch.argmax(pred[train_idx], dim=1)
    pred_val = torch.argmax(pred[val_idx], dim=1)
    pred_test = torch.argmax(pred[test_idx], dim=1)
    metrics = {
        "train_acc": get_metric(pred_train, labels[train_idx], "accuracy", average=average),
        "val_acc": get_metric(pred_val, labels[val_idx], "accuracy", average=average),
        "test_acc": get_metric(pred_test, labels[test_idx], "accuracy", average=average),
        "train_f1": get_metric(pred_train, labels[train_idx], "f1", average=average),
        "val_f1": get_metric(pred_val, labels[val_idx], "f1", average=average),
        "test_f1": get_metric(pred_test, labels[test_idx], "f1", average=average),
    }
    return metrics, val_loss, test_loss


def run_single_experiment(args, graph, cross_graph, text_feat, image_feat, labels, train_idx, val_idx, test_idx, device):
    model = TopSCAN(
        gnn_type=args.gnn_type,
        text_input_dim=text_feat.shape[1],
        image_input_dim=image_feat.shape[1],
        hidden_dim=args.n_hidden,
        num_classes=int((labels.max() + 1).item()),
        num_layers=args.n_layers,
        dropout=args.dropout,
        cma_num_heads=args.cma_num_heads,
        cma_head_dim=args.cma_head_dim,
        cma_dropout=args.cma_dropout,
        cma_attn_dropout=args.cma_attn_dropout,
        cma_norm_by=args.cma_norm_by,
        warmup_fusion=args.warmup_fusion,
        warmup_alpha=float(getattr(args, "warmup_alpha", 0.5) or 0.5),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=100, verbose=False, min_lr=args.min_lr
    )
    stopper = EarlyStopping(patience=args.early_stop_patience) if args.early_stop_patience else None

    best_val_acc = 0.0
    best_val_f1 = 0.0
    final_test_acc = 0.0
    final_test_f1 = 0.0
    best_epoch = 0
    best_metric = float("-inf")
    best_checkpoint_path = ""
    total_train_time = 0.0
    save_checkpoints = bool(getattr(args, "save_checkpoints", False))
    checkpoint_metric = str(getattr(args, "checkpoint_metric", "val_acc")).lower()
    run_index = int(getattr(args, "current_run_idx", 0))

    for epoch in range(1, args.n_epochs + 1):
        if args.warmup_epochs and epoch <= args.warmup_epochs:
            adjust_learning_rate(optimizer, args.lr, epoch, args.warmup_epochs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        train_loss, _ = train(
            model, graph, cross_graph, text_feat, image_feat, labels, train_idx, optimizer, args.label_smoothing
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_train_time += time.time() - t0

        if epoch % args.eval_steps == 0:
            metrics, val_loss, _ = evaluate(
                model,
                graph,
                cross_graph,
                text_feat,
                image_feat,
                labels,
                train_idx,
                val_idx,
                test_idx,
                label_smoothing=args.label_smoothing,
                average=args.average,
            )
            val_acc = metrics["val_acc"]
            val_f1 = metrics["val_f1"]
            test_acc = metrics["test_acc"]
            test_f1 = metrics["test_f1"]
            scheduler.step(val_loss)

            current_metric = val_acc if checkpoint_metric == "val_acc" else val_f1
            is_better = current_metric > best_metric
            if current_metric == best_metric and val_acc > best_val_acc:
                is_better = True

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                final_test_acc = test_acc
                final_test_f1 = test_f1

            if is_better:
                best_metric = current_metric
                best_epoch = epoch
                if save_checkpoints:
                    best_checkpoint_path = _save_best_checkpoint(
                        model=model,
                        args=args,
                        run_idx=run_index,
                        epoch=epoch,
                        best_metric=best_metric,
                        best_val_acc=best_val_acc,
                        best_val_f1=best_val_f1,
                        test_acc=test_acc,
                        test_f1=test_f1,
                    )

            if stopper and stopper.step(val_acc):
                break

            if args.verbose and epoch % args.log_every == 0:
                print(
                    f"Epoch {epoch:04d} | Loss {train_loss:.4f} | "
                    f"Val Acc {val_acc:.4f} | Test Acc {test_acc:.4f} | "
                    f"Val F1 {val_f1:.4f} | Test F1 {test_f1:.4f}"
                )

    model.eval()
    inference_times = []
    with torch.no_grad():
        for _ in range(3):
            _ = model(graph, cross_graph, text_feat, image_feat)
        for _ in range(10):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            _ = model(graph, cross_graph, text_feat, image_feat)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_times.append(time.time() - t0)
    avg_inference_time = float(np.mean(inference_times))

    return {
        "best_val_acc": best_val_acc,
        "best_val_f1": best_val_f1,
        "final_test_acc": final_test_acc,
        "final_test_f1": final_test_f1,
        "best_epoch": int(best_epoch),
        "train_time": total_train_time,
        "infer_time": avg_inference_time,
        "best_checkpoint_path": best_checkpoint_path,
    }


def get_args():
    parser = argparse.ArgumentParser()
    from topscan.utils.model_config import add_common_args, add_topscan_args

    add_common_args(parser)
    add_topscan_args(parser)
    return parser.parse_args()


def main():
    args = get_args()
    if not str(args.output_file).strip():
        args.output_file = "topscan_results.csv"
    if not str(args.data_name).strip():
        args.data_name = Path(args.graph_path).stem if args.graph_path else "dataset"

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")

    def _to_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        return obj

    graph, labels, train_idx, val_idx, test_idx = load_data(
        args.graph_path,
        train_ratio=float(getattr(args, "train_ratio", 0.6)),
        val_ratio=float(getattr(args, "val_ratio", 0.2)),
        name=str(args.data_name),
        fewshots=getattr(args, "fewshots", None),
    )
    if not graph.is_homogeneous or ("feat" in graph.ndata):
        pass
    if str(args.cross_graph_path).strip():
        cross_graph, _cl, _tr, _va, _te = load_data(
            args.cross_graph_path,
            train_ratio=float(getattr(args, "train_ratio", 0.6)),
            val_ratio=float(getattr(args, "val_ratio", 0.2)),
            name=str(args.data_name) + "_cross",
            fewshots=getattr(args, "fewshots", None),
        )
    else:
        cross_graph = graph

    graph = graph.to(device)
    cross_graph = cross_graph.to(device)
    labels = _to_tensor(labels)
    train_idx = _to_tensor(train_idx)
    val_idx = _to_tensor(val_idx)
    test_idx = _to_tensor(test_idx)

    text_feat = torch.from_numpy(np.load(args.text_feature).astype(np.float32)).to(device)
    visual_feat = torch.from_numpy(np.load(args.visual_feature).astype(np.float32)).to(device)

    if int(getattr(args, "grid_search", 0)) == 1:
        configs = build_grid_configs(args)
    else:
        configs = [
            {
                "gnn_type": args.gnn_type,
                "n_layers": int(args.n_layers),
                "n_hidden": int(args.n_hidden),
                "lr": float(args.lr),
                "dropout": float(args.dropout),
                "fusion": str(args.warmup_fusion),
                "warmup_alpha": normalize_alpha_by_fusion(args.warmup_fusion, args.warmup_alpha),
                "cma_num_heads": int(args.cma_num_heads),
                "cma_head_dim": int(args.cma_head_dim),
                "cma_attn_dropout": float(args.cma_attn_dropout),
                "cma_norm_by": str(args.cma_norm_by),
            }
        ]

    rows = []
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        args.gnn_type = cfg["gnn_type"]
        args.n_layers = int(cfg["n_layers"])
        args.n_hidden = int(cfg["n_hidden"])
        args.lr = float(cfg["lr"])
        args.dropout = float(cfg["dropout"])
        args.warmup_fusion = cfg["fusion"]
        if cfg["warmup_alpha"] is not None:
            args.warmup_alpha = float(cfg["warmup_alpha"])
        args.cma_num_heads = int(cfg["cma_num_heads"])
        args.cma_head_dim = int(cfg["cma_head_dim"])
        args.cma_attn_dropout = float(cfg["cma_attn_dropout"])
        args.cma_norm_by = str(cfg["cma_norm_by"])

        val_accs, test_accs, val_f1s, test_f1s, train_times, infer_times = [], [], [], [], [], []
        for run in range(args.n_runs):
            set_seed(args.seed + run)
            args.current_run_idx = run
            result = run_single_experiment(
                args,
                graph,
                cross_graph,
                text_feat,
                visual_feat,
                labels,
                train_idx,
                val_idx,
                test_idx,
                device,
            )
            val_accs.append(result["best_val_acc"])
            test_accs.append(result["final_test_acc"])
            val_f1s.append(result["best_val_f1"])
            test_f1s.append(result["final_test_f1"])
            train_times.append(result["train_time"])
            infer_times.append(result["infer_time"])

        rows.append(
            {
                "data_name": args.data_name,
                "gnn_type": args.gnn_type,
                "n_layers": args.n_layers,
                "n_hidden": args.n_hidden,
                "lr": args.lr,
                "dropout": args.dropout,
                "warmup_fusion": args.warmup_fusion,
                "warmup_alpha": float(getattr(args, "warmup_alpha", 0.0) or 0.0),
                "cma_num_heads": args.cma_num_heads,
                "cma_head_dim": args.cma_head_dim,
                "cma_attn_dropout": args.cma_attn_dropout,
                "cma_norm_by": args.cma_norm_by,
                "policy_mode": str(getattr(args, "policy_mode", "off")),
                "policy_k": int(getattr(args, "policy_k", 0)),
                "policy_q": float(getattr(args, "policy_q", 0.0)),
                "policy_top_m": int(getattr(args, "policy_top_m", 0)),
                "policy_top_p": float(getattr(args, "policy_top_p", 0.0)),
                "enhance_cross_graph": int(getattr(args, "enhance_cross_graph", 0)),
                "use_enhanced_graph_in_backbone": int(getattr(args, "use_enhanced_graph_in_backbone", 0)),
                "avg_val_acc": float(np.mean(val_accs)) if val_accs else 0.0,
                "avg_test_acc": float(np.mean(test_accs)) if test_accs else 0.0,
                "std_test_acc": float(np.std(test_accs)) if test_accs else 0.0,
                "avg_val_f1_macro": float(np.mean(val_f1s)) if val_f1s else 0.0,
                "avg_test_f1_macro": float(np.mean(test_f1s)) if test_f1s else 0.0,
                "std_test_f1_macro": float(np.std(test_f1s)) if test_f1s else 0.0,
                "avg_train_time_sec": float(np.mean(train_times)) if train_times else 0.0,
                "avg_infer_time_sec": float(np.mean(infer_times)) if infer_times else 0.0,
                "n_runs": int(args.n_runs),
            }
        )
        pd.DataFrame(rows).to_csv(output_file, index=False)
        print(
            f"[done] {args.gnn_type} layers={args.n_layers} hidden={args.n_hidden} "
            f"test_acc={float(np.mean(test_accs)):.4f}±{float(np.std(test_accs)):.4f} "
            f"test_f1={float(np.mean(test_f1s)):.4f}±{float(np.std(test_f1s)):.4f}"
        )

    pd.DataFrame(rows).to_csv(output_file, index=False)
    print(f"Saved results to: {output_file.resolve()}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
