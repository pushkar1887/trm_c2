from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import copy
import traceback

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
try:
    from adam_atan2 import AdamATan2
    ADAM_ATAN2_IMPORT_ERROR = None
except Exception as exc:
    AdamATan2 = None
    ADAM_ATAN2_IMPORT_ERROR = exc

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    dataloader_num_workers: int = 1
    dataloader_prefetch_factor: int = 8
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float
    optimizer: str = "auto"

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0 # when to start eval
    eval_save_outputs: List[str] = []

    ema: bool = False # use Exponential-Moving-Average
    ema_rate: float = 0.999 # EMA-rate
    freeze_weights: bool = False # If True, freeze weights and only learn the embeddings
    disable_compile: bool = False # Disable torch.compile/Inductor for environments without Triton
    terminal_log_every: int = 10 # Show selected train metrics in the terminal progress bar

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    arch_extra = config.arch.__pydantic_extra__ or {}
    c2_enabled = bool(arch_extra.get("c2_enabled", False))
    c2_mode = str(arch_extra.get("c2_mode", ""))
    c2_num_context = int(arch_extra.get("c2_num_context", 0)) if c2_enabled and c2_mode == "test_conditioned" else 0
    c2_visual_cache_path = arch_extra.get("c2_visual_cache_path") if bool(arch_extra.get("c2_visual_encoder", False)) else None
    c2_relmap = bool(arch_extra.get("c2_relmap", False))
    c2_frame_hint = bool(arch_extra.get("c2_frame_hint", False))
    dataset_kwargs = dict(kwargs)
    dataset_kwargs.setdefault("c2_relmap", c2_relmap)
    dataset_kwargs.setdefault("c2_frame_hint", c2_frame_hint)
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test)>0 and split=="test" else config.data_paths,
        rank=rank,
        num_replicas=world_size,
        c2_num_context=c2_num_context,
        c2_visual_cache_path=c2_visual_cache_path,
        **dataset_kwargs
    ), split=split)
    num_workers = int(config.dataloader_num_workers)
    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=bool(config.dataloader_pin_memory),
    )
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = int(config.dataloader_prefetch_factor)
        dataloader_kwargs["persistent_workers"] = bool(config.dataloader_persistent_workers)
    dataloader = DataLoader(**dataloader_kwargs)
    return dataloader, dataset.metadata


def create_dense_optimizer(params, config: PretrainConfig):
    optimizer_name = config.optimizer.lower()
    if optimizer_name not in {"auto", "adam_atan2", "adamw"}:
        raise ValueError(
            f"Unknown optimizer={config.optimizer!r}; use auto, adam_atan2, or adamw."
        )

    use_adamw = optimizer_name == "adamw"
    if optimizer_name == "adam_atan2" and AdamATan2 is None:
        raise RuntimeError(
            "optimizer=adam_atan2 was requested, but adam_atan2 could not be "
            f"imported. Original error: {ADAM_ATAN2_IMPORT_ERROR!r}"
        )
    if optimizer_name == "auto" and AdamATan2 is None:
        use_adamw = True
        print(
            "[optimizer] AdamATan2 unavailable; falling back to torch.optim.AdamW. "
            f"Original error: {ADAM_ATAN2_IMPORT_ERROR!r}"
        )

    if use_adamw:
        print("[optimizer] dense optimizer = AdamW")
        return torch.optim.AdamW(
            params,
            lr=0,  # Needs to be set by scheduler
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )

    print("[optimizer] dense optimizer = AdamATan2")
    return AdamATan2(
        params,
        lr=0,  # Needs to be set by scheduler
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
    )


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if config.disable_compile or "DISABLE_COMPILE" in os.environ:
            print("[compile] disabled")
        else:
            model = torch.compile(model)  # type: ignore

        # Load checkpoint
        if rank == 0:
            load_checkpoint(model, config)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    if config.arch.puzzle_emb_ndim == 0:
        optimizers = [
            create_dense_optimizer(model.parameters(), config)
        ]
        optimizer_lrs = [
            config.lr
        ]
    elif config.freeze_weights:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr
        ]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            ),
            create_dense_optimizer(model.parameters(), config)
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr,
            config.lr
        ]

    return model, optimizers, optimizer_lrs

def mix_weights_direct(device, alpha, net, nets):
    sd = []
    for i in range(len(nets)):
        sd += [nets[i].state_dict()]
    sd_alpha = {}
    for k in sd[0].keys():
        comb_net = alpha[0]*sd[0][k].to(device)
        for i in range(1,len(nets)):
            comb_net += alpha[i]*sd[i][k].to(device)
        sd_alpha[k] =  comb_net
    net.load_state_dict(sd_alpha)
    return net

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    # FIXME: Only saved model.
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def _prepare_evidence_schema_state(
    state_dict: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
    schema_name: str,
    *,
    allow_legacy: bool,
) -> list[str]:
    """Validate evidence semantics and make a legacy warm-start unambiguous.

    Equal widths do not imply equal evidence meanings. Fingerprinted checkpoints must match
    exactly. Explicitly accepted pre-fingerprint checkpoints retain unrelated TRM weights but
    discard every learned tensor whose columns encode the new evidence schema.
    """
    if schema_name not in model_state:
        return []
    if schema_name in state_dict:
        if not torch.equal(
            state_dict[schema_name].detach().cpu(), model_state[schema_name].detach().cpu()
        ):
            raise RuntimeError(
                "Checkpoint evidence schema is semantically incompatible with the active "
                "ordered evidence layout. Tensor widths alone are not a valid migration.")
        return []
    if not allow_legacy:
        raise RuntimeError(
            "Checkpoint predates evidence-schema fingerprints. Set "
            "arch.c2_allow_legacy_evidence_schema=true only for an audited legacy warm-start.")

    inner_prefix = schema_name[:-len("evidence_schema_fingerprint")]
    semantic_prefixes = (
        f"{inner_prefix}color_evidence_proj.",
        f"{inner_prefix}color_head_mlp_in.",
        f"{inner_prefix}color_head_mlp_out.",
        f"{inner_prefix}rule_factor_proj.",
        f"{inner_prefix}pairdelta_input_encoder.spatial_mlp.",
    )
    dropped = sorted(
        key for key in state_dict
        if any(key.startswith(prefix) for prefix in semantic_prefixes)
    )
    for key in dropped:
        del state_dict[key]
    # Install the active schema buffer so assign=True cannot retain an absent/ambiguous contract.
    state_dict[schema_name] = model_state[schema_name].detach().clone()
    return dropped


def load_checkpoint(model: nn.Module, config: PretrainConfig):
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")

        # Load state dict
        state_dict = torch.load(config.load_checkpoint, map_location="cuda")
        model_state = model.state_dict()

        model_uses_compile_prefix = any(k.startswith("_orig_mod.") for k in model_state)
        state_uses_compile_prefix = any(k.startswith("_orig_mod.") for k in state_dict)
        if state_uses_compile_prefix and not model_uses_compile_prefix:
            state_dict = {
                (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
                for k, v in state_dict.items()
            }
        elif model_uses_compile_prefix and not state_uses_compile_prefix:
            state_dict = {f"_orig_mod.{k}": v for k, v in state_dict.items()}

        def _state_key(name: str) -> str:
            return f"_orig_mod.{name}" if model_uses_compile_prefix else name

        schema_name = _state_key("model.inner.evidence_schema_fingerprint")
        legacy_schema = schema_name in model_state and schema_name not in state_dict
        legacy_dropped = _prepare_evidence_schema_state(
            state_dict,
            model_state,
            schema_name,
            allow_legacy=bool(getattr(config.arch, "c2_allow_legacy_evidence_schema", False)),
        )
        if legacy_schema:
            print(
                "[checkpoint] LEGACY evidence schema accepted; reset "
                f"{len(legacy_dropped)} semantic-consumer tensor(s).")

        # Resize and reset puzzle emb if needed
        puzzle_emb_name = _state_key("model.inner.puzzle_emb.weights")
        expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
        if puzzle_emb_name in state_dict:
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                print(f"Resetting puzzle embedding as shape is different. Found {puzzle_emb.shape}, Expected {expected_shape}")
                # Re-initialize using mean
                state_dict[puzzle_emb_name] = (
                    torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
                )

        grid_encoder_name = _state_key("model.inner.grid_encoder.embed_tokens.embedding_weight")
        embed_tokens_name = _state_key("model.inner.embed_tokens.embedding_weight")
        if grid_encoder_name in model_state and grid_encoder_name not in state_dict and embed_tokens_name in state_dict:
            state_dict[grid_encoder_name] = state_dict[embed_tokens_name]

        lm_head_name = _state_key("model.inner.lm_head.weight")
        color_head_name = _state_key("model.inner.color_head.weight")
        structure_head_name = _state_key("model.inner.structure_head.weight")
        if lm_head_name in state_dict:
            lm_head = state_dict[lm_head_name]
            if (
                color_head_name in model_state
                and color_head_name not in state_dict
                and lm_head.ndim == 2
                and lm_head.shape[0] >= 12
            ):
                color_shape = model_state[color_head_name].shape
                color_head = torch.zeros(color_shape, dtype=lm_head.dtype, device=lm_head.device)
                hidden_cols = min(lm_head.shape[1], color_shape[1])
                color_rows = min(10, color_shape[0])
                color_head[:color_rows, :hidden_cols] = lm_head[2:2 + color_rows, :hidden_cols]
                state_dict[color_head_name] = color_head
                print("[checkpoint] Warm-started color_head from lm_head colour rows.")
            if (
                color_head_name in model_state
                and color_head_name in state_dict
                and state_dict[color_head_name].shape != model_state[color_head_name].shape
                and state_dict[color_head_name].ndim == 2
                and model_state[color_head_name].ndim == 2
            ):
                old_color_head = state_dict[color_head_name]
                color_shape = model_state[color_head_name].shape
                color_head = torch.zeros(color_shape, dtype=old_color_head.dtype, device=old_color_head.device)
                rows = min(color_shape[0], old_color_head.shape[0])
                cols = min(color_shape[1], old_color_head.shape[1])
                color_head[:rows, :cols] = old_color_head[:rows, :cols]
                state_dict[color_head_name] = color_head
                print("[checkpoint] Expanded color_head warm-start with zero-init new feature columns.")
            if (
                structure_head_name in model_state
                and structure_head_name not in state_dict
                and lm_head.ndim == 2
                and lm_head.shape[0] >= 12
            ):
                structure_shape = model_state[structure_head_name].shape
                structure_head = torch.zeros(structure_shape, dtype=lm_head.dtype, device=lm_head.device)
                hidden_cols = min(lm_head.shape[1], structure_shape[1])
                structure_head[0, :hidden_cols] = lm_head[0, :hidden_cols]
                structure_head[1, :hidden_cols] = lm_head[1, :hidden_cols]
                structure_head[2, :hidden_cols] = lm_head[2:12, :hidden_cols].mean(dim=0)
                state_dict[structure_head_name] = structure_head
                print("[checkpoint] Warm-started structure_head from lm_head PAD/EOS/colour rows.")

        try:
            model.load_state_dict(state_dict, assign=True)
        except RuntimeError as exc:
            expected_missing_fragments = (
                "model.inner.evidence_schema_fingerprint",
                "model.inner.relmap_proj.",
                "model.inner.frame_embed.",
                "model.inner.rule_hyp_embed.",
                "model.inner.structure_head.",
                "model.inner.structure_relmap_proj.",
                "model.inner.structure_outside_proj.",
                "model.inner.structure_eos_proj.",
                "model.inner.structure_pairdelta_proj.",  # D9 (File #5): fresh zero-init lever

                "model.inner.color_head.",
                "model.inner.color_evidence_proj.",  # FIX A: evidence columns split out of color_head
                "model.inner.color_head_mlp_in.",
                "model.inner.color_head_mlp_out.",
                "model.inner.quarantine_",          # PID-quarantined candidate head (lin + mlp_in/out)
                "model.inner.c2_demo_relmap_proj.",
                "model.inner.pairdelta_input_encoder.",
                "model.inner.delta_rule_input_proj.",
                "model.inner.rule_factor_proj.",
                "model.inner.color_residual_head.",
                "model.inner.color_residual_gate",
                "model.inner.grid_encoder.visual_encoder.",
                "model.inner.grid_encoder.visual_gate",
                "model.inner.visual_rule_adapter.",
                "model.inner.shape_h_head.",
                "model.inner.shape_w_head.",
                "model.inner.c2.",
                "model.inner.pid_task_gate",
                "model.inner.pid_task_modulator.",
                "model.inner.delta_rule_encoder.",
                "model.inner.delta_rule_proj.",
                "model.inner.delta_rule_gate",
                "model.inner.delta_rule_logit_fuse.",
                "model.inner.delta_rule_logit_head.",
                "model.inner.delta_rule_cell_gate.",
                "model.inner.delta_rule_slot_attn.",
                "model.inner.delta_rule_struct_head.",
                "model.inner.delta_rule_color_head.",
                "model.inner.color_transition_bank.",
                "model.inner.c2_color_prior_gate",
                "model.inner.delta_rule_prior_proj.",
                "model.inner.c2_copy_structure_gate",
                "model.inner.color_repair_head.",
                "model.inner.c2_shape_canvas_gate",        # S4 canvas builder strength
                "model.inner.color_force_head.",           # S3 colour-forcing keystone head
                "model.inner.rule_bus.",                   # S2 rule bus (floor/solver/struct projections)
            )
            # Drop tensors whose shape changed (e.g. relmap_proj after the directional-channel split
            # 10->13): assign=True would otherwise replace the fresh module with an incompatible tensor
            # and crash the forward. Dropping makes them MISSING -> re-init (the allowlist permits it).
            for k in [k for k in list(state_dict) if k in model_state and state_dict[k].shape != model_state[k].shape]:
                del state_dict[k]
            state_keys = set(state_dict)
            model_keys = set(model_state)
            missing = sorted(model_keys - state_keys)
            unexpected = sorted(state_keys - model_keys)
            bad_missing = [
                k for k in missing
                if not any(fragment in k for fragment in expected_missing_fragments)
            ]
            if bad_missing or unexpected:
                raise exc

            print("[checkpoint] Compatible partial warm-start.")
            print(f"[checkpoint] Missing newly initialized keys: {len(missing)}")
            for key in missing[:20]:
                print(f"  - {key}")
            if len(missing) > 20:
                print(f"  ... {len(missing) - 20} more")
            model.load_state_dict(state_dict, strict=False, assign=True)


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )


def terminal_metric_postfix(metrics: dict[str, Any]) -> dict[str, float]:
    preferred = (
        "train/loss",
        "train/exact_accuracy",
        "train/accuracy",
        "train/c2_delta_lodo_loss",
        "train/c2_delta_changed_acc",
        "train/c2_delta_color_acc",
        "train/c2_delta_pad_acc",
        "train/c2_delta_eos_acc",
        "train/lr",
    )
    postfix: dict[str, float] = {}
    for key in preferred:
        value = metrics.get(key)
        if value is None:
            continue
        if torch.is_tensor(value):
            value = value.detach().float().cpu().item()
        postfix[key.removeprefix("train/")] = float(value)
    if postfix:
        return postfix

    for key, value in list(metrics.items())[:4]:
        if torch.is_tensor(value):
            value = value.detach().float().cpu().item()
        try:
            postfix[key.removeprefix("train/")] = float(value)
        except (TypeError, ValueError):
            continue
    return postfix



def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths =config.data_paths_test if len(config.data_paths_test)>0 else config.data_paths
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )  # type: ignore
            evaluators.append(cls)

    return evaluators

def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    ((1 / global_batch_size) * loss).backward()

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
            
    # Apply optimizer
    lr_this_step = None    
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
            
        optim.step()
        optim.zero_grad()

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics

def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
):
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds = {}

        metric_keys = []
        metric_values = None

        carry = None
        processed_batches = 0
        
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")
            
            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1

                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda"
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del metrics

        # concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_preds):
            # Each rank save predictions independently
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
            )

        del save_preds

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")
            
        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
            # Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            # Run and log
            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")
                
        if rank == 0:
            print("All evaluators completed!")

    return reduced_metrics

def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
        # CPU GLOO process group
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    try:
        eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    except Exception as e:
        # Do NOT swallow config typos / import bugs as "no eval data": log the real cause. Narrow to
        # Exception so KeyboardInterrupt / SystemExit still propagate.
        print(f"NO EVAL DATA FOUND: {type(e).__name__}: {e}")
        traceback.print_exc()
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except Exception as e:
        print(f"No evaluator found: {type(e).__name__}: {e}")
        traceback.print_exc()
        evaluators = []

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar = None
    ema_helper = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)
    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Training Loop
    for _iter_id in range(total_iters):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        if RANK == 0:
            print("TRAIN")
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
                if config.terminal_log_every > 0 and train_state.step % config.terminal_log_every == 0:
                    progress_bar.set_postfix(terminal_metric_postfix(metrics))  # type: ignore
            if config.ema:
                ema_helper.update(train_state.model)

        if _iter_id >= config.min_eval_interval:
            ############ Evaluation
            if RANK == 0:
                print("EVALUATE")
            if config.ema:
                print("SWITCH TO EMA")
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state
            train_state_eval.model.eval()
            metrics = evaluate(config, 
                train_state_eval, 
                eval_loader, 
                eval_metadata, 
                evaluators,
                rank=RANK, 
                world_size=WORLD_SIZE,
                cpu_group=CPU_PROCESS_GROUP)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                
            ############ Checkpointing
            if RANK == 0:
                print("SAVE CHECKPOINT")
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                save_train_state(config, train_state_eval)

            if config.ema:
                del train_state_eval

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
