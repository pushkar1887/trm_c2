import os
import json
import hashlib
from typing import Tuple, List, Dict, Optional
import numpy as np
import pydantic

import torch
from torch.utils.data import IterableDataset, get_worker_info

from models.losses import IGNORE_LABEL_ID
from dataset.build_arc_dataset import ARCMaxGridSize, arc_grid_to_np, np_grid_to_seq_translational_augment
from dataset.common import PuzzleDatasetMetadata

from argdantic import ArgParser
from pydantic import BaseModel

def _sample_batch(rng: np.random.Generator, group_order: np.ndarray, puzzle_indices: np.ndarray, group_indices: np.ndarray, start_index: int, global_batch_size: int):
    # Pack examples into a full batch
    batch = []
    batch_puzzle_indices = []
    current_size = 0

    while (start_index < group_order.size) and (current_size < global_batch_size):
        # Pick a group and a puzzle from that group
        group_id = group_order[start_index]
        puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
        start_index += 1

        # Get range of the puzzle
        puzzle_start = puzzle_indices[puzzle_id]
        puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)

        append_size = min(puzzle_size, global_batch_size - current_size)

        # Put into batch
        batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))
        batch.append(puzzle_start + rng.choice(puzzle_size, append_size, replace=False))

        current_size += append_size

    return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)


def _derive_target_hw_from_label_tokens(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback H/W recovery for older datasets without saved shape arrays."""
    flat = labels.reshape(labels.shape[0], -1)
    height = np.ones((flat.shape[0],), dtype=np.int32)
    width = np.ones((flat.shape[0],), dtype=np.int32)

    for row, seq in enumerate(flat):
        if not np.any(seq != IGNORE_LABEL_ID):
            continue
        seq = np.where(seq == IGNORE_LABEL_ID, 0, seq)
        grid_size = int(np.sqrt(seq.shape[0]))
        assert grid_size * grid_size == seq.shape[0]
        grid = seq.reshape(grid_size, grid_size)

        # Existing train arrays may be translationally augmented, so top-left
        # crop is invalid. The valid-token bounding box recovers raw canvas H/W.
        valid = (grid >= 2) & (grid <= 11)
        if not valid.any():
            continue
        rows, cols = np.where(valid)
        h = int(rows.max() - rows.min() + 1)
        w = int(cols.max() - cols.min() + 1)
        assert 1 <= h <= ARCMaxGridSize
        assert 1 <= w <= ARCMaxGridSize
        height[row] = h
        width[row] = w

    return height, width


class PuzzleDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_paths: List[str]
    global_batch_size: int
    test_set_mode: bool
    epochs_per_iter: int  # Batch X epochs in an iteration to reduce overhead.
    rank: int
    num_replicas: int
    c2_num_context: int = 0
    c2_visual_cache_path: Optional[str] = None
    c2_relmap: bool = False
    c2_frame_hint: bool = False


class VisualFeatureCache:
    """Lightweight lookup for frozen visual feature shards keyed by token SHA1."""

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        manifest_path = os.path.join(cache_path, "manifest.jsonl")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Visual cache manifest not found: {manifest_path}")

        self.index: Dict[bytes, Tuple[str, int]] = {}
        self.feature_shape: Optional[Tuple[int, int]] = None
        self.cache_level: Optional[str] = None
        self.model_name: Optional[str] = None
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                shape = row.get("shape")
                if (
                    not isinstance(shape, list)
                    or len(shape) != 2
                    or int(shape[0]) != 256
                    or int(shape[1]) <= 0
                ):
                    raise ValueError(f"Unsupported cached feature shape in manifest: {shape}")
                if row.get("dtype") != "float16":
                    raise ValueError(f"Unsupported cached feature dtype in manifest: {row.get('dtype')}")
                shape_tuple = (int(shape[0]), int(shape[1]))
                if self.feature_shape is None:
                    self.feature_shape = shape_tuple
                    self.cache_level = row.get("cache_level")
                    self.model_name = row.get("model")
                elif self.feature_shape != shape_tuple:
                    raise ValueError(
                        f"Mixed cached feature shapes in manifest: first={self.feature_shape} row={shape_tuple}"
                    )
                self.index[bytes.fromhex(row["sha1"])] = (row["feature_file"], int(row["row"]))
        if not self.index:
            raise ValueError(f"Visual cache manifest is empty: {manifest_path}")
        assert self.feature_shape is not None

        self._shards: Dict[str, torch.Tensor] = {}

    @staticmethod
    def hash_tokens(tokens: np.ndarray) -> bytes:
        arr = np.asarray(tokens, dtype=np.uint8).reshape(-1)
        return hashlib.sha1(arr.tobytes()).digest()

    def _load_shard(self, feature_file: str) -> torch.Tensor:
        shard = self._shards.get(feature_file)
        if shard is None:
            path = os.path.join(self.cache_path, feature_file)
            try:
                shard = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                shard = torch.load(path, map_location="cpu")
            if not isinstance(shard, torch.Tensor):
                raise TypeError(f"Expected tensor shard in {path}, got {type(shard)!r}")
            if shard.ndim != 3 or tuple(shard.shape[1:]) != self.feature_shape:
                raise ValueError(f"Bad feature shard shape in {path}: {tuple(shard.shape)}")
            if shard.dtype != torch.float16:
                raise ValueError(f"Bad feature shard dtype in {path}: {shard.dtype}")
            self._shards[feature_file] = shard
        return shard

    def lookup_one(self, tokens: np.ndarray) -> np.ndarray:
        digest = self.hash_tokens(tokens)
        item = self.index.get(digest)
        if item is None:
            raise KeyError(
                "Visual feature cache miss for token grid sha1="
                f"{digest.hex()} in {self.cache_path}"
            )
        feature_file, row = item
        shard = self._load_shard(feature_file)
        return shard[row].numpy()

    def lookup_batch(self, tokens: np.ndarray) -> np.ndarray:
        arr = np.asarray(tokens)
        assert arr.shape[-1] == 900, f"Expected token grids ending in 900, got {arr.shape}"
        flat = arr.reshape(-1, arr.shape[-1])
        features = [self.lookup_one(row) for row in flat]
        assert self.feature_shape is not None
        return np.stack(features, axis=0).reshape(*arr.shape[:-1], *self.feature_shape)

class PuzzleDataset(IterableDataset):
    def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split

        # Merge multiple metadata
        prev_seq_len = None
        prev_vocab_size = None
        prev_pad_id = None
        prev_ignore_label_id = None
        prev_blank_identifier_id = None
        prev_sets = None
        prev_num_identifiers = None
        mean_puzzle_examples = 0
        total_puzzles = 0
        total_groups = 0
        num_identifiers = 0
        for dataset_path in config.dataset_paths:
            current_metadata = self._load_metadata(dataset_path)
            if prev_seq_len is None:
                prev_seq_len = current_metadata.seq_len
                prev_vocab_size = current_metadata.vocab_size
                prev_pad_id = current_metadata.pad_id
                prev_ignore_label_id = current_metadata.ignore_label_id
                prev_blank_identifier_id = current_metadata.blank_identifier_id
                prev_sets = current_metadata.sets
                prev_num_identifiers = current_metadata.num_puzzle_identifiers
            else:
                assert prev_seq_len == current_metadata.seq_len
                assert prev_vocab_size == current_metadata.vocab_size
                assert prev_pad_id == current_metadata.pad_id
                assert prev_ignore_label_id == current_metadata.ignore_label_id
                assert prev_blank_identifier_id == current_metadata.blank_identifier_id
                assert prev_sets == current_metadata.sets
                assert prev_num_identifiers == current_metadata.num_puzzle_identifiers
            mean_puzzle_examples += current_metadata.mean_puzzle_examples*current_metadata.total_puzzles
            total_puzzles += current_metadata.total_puzzles
            total_groups += current_metadata.total_groups
            num_identifiers += current_metadata.num_puzzle_identifiers
        mean_puzzle_examples = mean_puzzle_examples / total_puzzles

        self.metadata = PuzzleDatasetMetadata(
            seq_len=prev_seq_len,
            vocab_size=prev_vocab_size,
            pad_id=prev_pad_id,
            ignore_label_id=prev_ignore_label_id,
            blank_identifier_id=prev_blank_identifier_id,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=mean_puzzle_examples,
            total_puzzles=total_puzzles,
            sets=prev_sets
        )

        # Checks
        assert self.config.global_batch_size % self.config.num_replicas == 0, f"Global batch size {self.config.global_batch_size} must be multiples of nodes {self.config.num_replicas}."
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas

        # State
        self._data = None
        self._iters = 0
        self._visual_cache: Optional[VisualFeatureCache] = None

    def _load_metadata(self, dataset_path) -> PuzzleDatasetMetadata:
        with open(os.path.join(dataset_path, self.split, "dataset.json"), "r") as f:
            return PuzzleDatasetMetadata(**json.load(f))

    def _lazy_load_dataset(self):
        if self._data is not None:
            return

        if self.config.c2_visual_cache_path is not None:
            self._visual_cache = VisualFeatureCache(self.config.c2_visual_cache_path)

        field_mmap_modes = {
            "inputs": "r",
            "labels": "r",

            # Keep indices in memory
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None
        }

        # Load data
        self._data = {}
        for set_name in self.metadata.sets: # Load subset
            for i, dataset_path in enumerate(self.config.dataset_paths):
                if i > 0:
                    set_name_ = set_name + str(i)
                else:
                    set_name_ = set_name
                self._data[set_name_] = {
                    field_name: np.load(os.path.join(dataset_path, self.split, f"{set_name}__{field_name}.npy"), mmap_mode=mmap_mode)
                    for field_name, mmap_mode in field_mmap_modes.items()
                }
                target_height_path = os.path.join(dataset_path, self.split, f"{set_name}__target_height.npy")
                target_width_path = os.path.join(dataset_path, self.split, f"{set_name}__target_width.npy")
                if os.path.exists(target_height_path) or os.path.exists(target_width_path):
                    if not (os.path.exists(target_height_path) and os.path.exists(target_width_path)):
                        raise FileNotFoundError("target_height and target_width arrays must be present together")
                    self._data[set_name_]["target_height"] = np.load(target_height_path, mmap_mode=None)
                    self._data[set_name_]["target_width"] = np.load(target_width_path, mmap_mode=None)
                if self.config.c2_num_context > 0 and self.split == "test":
                    self._data[set_name_]["_test_context"] = self._load_test_context(dataset_path)

    def _load_test_context(self, dataset_path: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        identifiers_path = os.path.join(dataset_path, "identifiers.json")
        test_puzzles_path = os.path.join(dataset_path, "test_puzzles.json")
        if not (os.path.exists(identifiers_path) and os.path.exists(test_puzzles_path)):
            return {}

        with open(identifiers_path, "r", encoding="utf-8") as f:
            identifiers = json.load(f)
        with open(test_puzzles_path, "r", encoding="utf-8") as f:
            test_puzzles = json.load(f)

        context = {}
        for identifier_id, puzzle_name in enumerate(identifiers):
            puzzle = test_puzzles.get(puzzle_name)
            if puzzle is None:
                continue

            inputs = []
            outputs = []
            for example in puzzle.get("train", []):
                inp, out = np_grid_to_seq_translational_augment(
                    arc_grid_to_np(example["input"]),
                    arc_grid_to_np(example["output"]),
                    do_translation=False,
                )
                inputs.append(inp)
                outputs.append(out)

            if inputs:
                context[identifier_id] = (np.stack(inputs, axis=0), np.stack(outputs, axis=0))
        return context

    def _empty_context(self, batch_size: int):
        context_shape = (batch_size, self.config.c2_num_context, self.metadata.seq_len)
        return {
            "context_inputs": np.full(context_shape, self.metadata.pad_id, dtype=np.int32),
            "context_outputs": np.full(context_shape, self.metadata.pad_id, dtype=np.int32),
            "context_mask": np.zeros((batch_size, self.config.c2_num_context), dtype=np.int32),
        }

    def _context_from_flat_examples(
        self,
        dataset,
        batch_indices: np.ndarray,
        batch_puzzle_indices: np.ndarray,
        rng: np.random.Generator,
    ):
        context = self._empty_context(batch_indices.size)
        for row, (example_index, puzzle_index) in enumerate(zip(batch_indices, batch_puzzle_indices)):
            puzzle_start = int(dataset["puzzle_indices"][puzzle_index])
            puzzle_end = int(dataset["puzzle_indices"][puzzle_index + 1])
            candidates = np.arange(puzzle_start, puzzle_end, dtype=np.int32)
            if candidates.size > 1:
                candidates = candidates[candidates != int(example_index)]
            if candidates.size == 0:
                continue

            take = min(self.config.c2_num_context, candidates.size)
            chosen = rng.choice(candidates, size=take, replace=False)
            context["context_inputs"][row, :take] = dataset["inputs"][chosen]
            context["context_outputs"][row, :take] = dataset["labels"][chosen]
            context["context_mask"][row, :take] = 1
        return context

    def _context_from_test_puzzles(self, dataset, batch_puzzle_indices: List[int]):
        context = self._empty_context(len(batch_puzzle_indices))
        lookup = dataset.get("_test_context", {})
        for row, puzzle_index in enumerate(batch_puzzle_indices):
            identifier_id = int(dataset["puzzle_identifiers"][puzzle_index])
            puzzle_context = lookup.get(identifier_id)
            if puzzle_context is None:
                continue
            inputs, outputs = puzzle_context
            take = min(self.config.c2_num_context, inputs.shape[0])
            context["context_inputs"][row, :take] = inputs[:take]
            context["context_outputs"][row, :take] = outputs[:take]
            context["context_mask"][row, :take] = 1
        return context


    def _attach_visual_features(self, batch_data: Dict[str, np.ndarray]) -> None:
        if self._visual_cache is None:
            return
        batch_data["input_visual_features"] = self._visual_cache.lookup_batch(batch_data["inputs"])
        if "context_inputs" in batch_data:
            batch_data["context_input_visual_features"] = self._visual_cache.lookup_batch(batch_data["context_inputs"])
            batch_data["context_output_visual_features"] = self._visual_cache.lookup_batch(batch_data["context_outputs"])

    def _collate_batch(self, batch):
        # Convert dtype
        float_keys = {
            "input_visual_features",
            "context_input_visual_features",
            "context_output_visual_features",
        }
        batch = {
            k: (v.astype(np.float16) if k in float_keys else v.astype(np.int32))
            for k, v in batch.items()
        }

        if "target_height" not in batch or "target_width" not in batch:
            target_height, target_width = _derive_target_hw_from_label_tokens(batch["labels"])
            batch["target_height"] = target_height
            batch["target_width"] = target_width

        assert np.all((batch["target_height"] >= 1) & (batch["target_height"] <= ARCMaxGridSize))
        assert np.all((batch["target_width"] >= 1) & (batch["target_width"] <= ARCMaxGridSize))

        # Convert ignore label IDs
        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        # Pad
        if batch["puzzle_identifiers"].size < self.local_batch_size:
            pad_size = self.local_batch_size - batch["puzzle_identifiers"].size
            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata.blank_identifier_id,
                "context_inputs": self.metadata.pad_id,
                "context_outputs": self.metadata.pad_id,
                "context_mask": 0,
                "target_height": 1,
                "target_width": 1,
                "input_visual_features": 0,
                "context_input_visual_features": 0,
                "context_output_visual_features": 0,
            }
            batch = {k: np.pad(v, ((0, pad_size), ) + ((0, 0), ) * (v.ndim - 1), constant_values=pad_values[k]) for k, v in batch.items()}

        # To tensor
        tensors = {k: torch.from_numpy(v) for k, v in batch.items()}
        tensors["target_height"] = tensors["target_height"].long()
        tensors["target_width"] = tensors["target_width"].long()
        # Pre-compute relational maps on CPU (B3: avoids O(L²) GPU recompute every forward step).
        if getattr(self.config, "c2_relmap", False):
            try:
                from models.recursive_reasoning.object_bank import relational_maps
                _side = int(np.sqrt(tensors["inputs"].shape[-1]))
                with torch.no_grad():
                    tensors["rel_maps"] = relational_maps(tensors["inputs"], side=_side)
                    if "context_inputs" in tensors:
                        B, C, L = tensors["context_inputs"].shape
                        ctx = tensors["context_inputs"].view(B * C, L)
                        ctx_maps = relational_maps(ctx, side=_side)
                        tensors["context_rel_maps"] = ctx_maps.view(B, C, L, -1)
                        # §15.2-A: support-OUTPUT maps too, so C2's demo features carry output-side geometry
                        # (the half input maps cannot show). Consumed only when c2_relmap_demos is on.
                        if "context_outputs" in tensors:
                            cout = tensors["context_outputs"].view(B * C, L)
                            cout_maps = relational_maps(cout, side=_side)
                            tensors["context_output_rel_maps"] = cout_maps.view(B, C, L, -1)
            except Exception as e:
                import warnings
                warnings.warn(f"relational_maps dataloader precompute failed: {e}")
                raise e
        # Lane-B rule-hypothesis HINT: the verified deterministic rearrange-FRAME family per task (0=none).
        # Precomputed here (CPU, no-grad) like rel_maps; consumed input-side via a zero-init embedding.
        if getattr(self.config, "c2_frame_hint", False) and "context_inputs" in tensors:
            try:
                from models.recursive_reasoning.object_rule_bank import task_frame_label
                _side = int(np.sqrt(tensors["inputs"].shape[-1]))
                B, C, L = tensors["context_inputs"].shape
                labels = torch.zeros(B, dtype=torch.long)
                with torch.no_grad():
                    for b in range(B):
                        labels[b] = task_frame_label(tensors["context_inputs"][b],
                                                     tensors["context_outputs"][b], _side)
                tensors["frame_label"] = labels
            except Exception as e:
                import warnings
                warnings.warn(f"frame_label dataloader precompute failed: {e}")
        return tensors
    
    def _iter_test(self):
        for set_i, (set_name, dataset) in enumerate(self._data.items()):  # type: ignore
            total_examples = len(dataset["inputs"])

            # Load examples one by one
            start_index = 0
            while start_index < total_examples:
                # Compute indices
                end_index = min(total_examples, start_index + self.config.global_batch_size)
                
                local_start = start_index + self.config.rank * self.local_batch_size
                local_end   = min(start_index + (self.config.rank + 1) * self.local_batch_size, end_index)
                
                # Get batch of examples, and also puzzle IDs
                puzzle_indices = []
                puzzle_index = np.searchsorted(dataset["puzzle_indices"], local_start, side="right") - 1
                for i in range(local_start, local_end):
                    while puzzle_index + 1 < len(dataset["puzzle_indices"]) and i >= dataset["puzzle_indices"][puzzle_index + 1]:
                        puzzle_index += 1

                    puzzle_indices.append(puzzle_index)
                
                batch_data = {
                    "inputs": dataset["inputs"][local_start: local_end],
                    "labels": dataset["labels"][local_start: local_end],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_indices]
                }
                if "target_height" in dataset and "target_width" in dataset:
                    batch_data["target_height"] = dataset["target_height"][local_start: local_end]
                    batch_data["target_width"] = dataset["target_width"][local_start: local_end]
                if self.config.c2_num_context > 0:
                    batch_data.update(self._context_from_test_puzzles(dataset, puzzle_indices))
                self._attach_visual_features(batch_data)
                batch = self._collate_batch(batch_data)

                yield set_name, batch, end_index - start_index
                
                # Advance to next batch
                start_index += self.config.global_batch_size

    def _iter_train(self):
        for set_name, dataset in self._data.items():  # type: ignore
            # Increase epoch count
            self._iters += 1

            # Randomly shuffle groups
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))

            group_order = np.concatenate([rng.permutation(dataset["group_indices"].size - 1) for _i in range(self.config.epochs_per_iter)])
            start_index = 0
            
            while start_index < group_order.size:
                start_index, batch_indices, batch_puzzle_indices = _sample_batch(
                    rng,
                    group_order=group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    start_index=start_index,
                    global_batch_size=self.config.global_batch_size,
                )

                # Select current rank and collate
                global_effective_batch_size = batch_puzzle_indices.size  # Global effective batch size, excluding pads

                # Drop last batch
                if global_effective_batch_size < self.config.global_batch_size:
                    break

                batch_indices        = batch_indices       [self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                batch_puzzle_indices = batch_puzzle_indices[self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                batch_data = {
                    "inputs": dataset["inputs"][batch_indices],
                    "labels": dataset["labels"][batch_indices],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][batch_puzzle_indices]
                }
                if "target_height" in dataset and "target_width" in dataset:
                    batch_data["target_height"] = dataset["target_height"][batch_indices]
                    batch_data["target_width"] = dataset["target_width"][batch_indices]
                if self.config.c2_num_context > 0:
                    batch_data.update(self._context_from_flat_examples(dataset, batch_indices, batch_puzzle_indices, rng))
                self._attach_visual_features(batch_data)
                batch = self._collate_batch(batch_data)

                yield set_name, batch, global_effective_batch_size
                
    def __iter__(self):
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1, "Multithreaded data loading is not currently supported."
        
        self._lazy_load_dataset()
        
        # Iterate using specified mode
        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()
