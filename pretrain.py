"""
This script is adapted from TinyLlama:
https://github.com/jzhang38/TinyLlama/blob/main/pretrain/tinyllama.py
"""

import math
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Tuple, Union

import lightning as L
import torch
import torch.nn as nn
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.throughput import ThroughputMonitor, measure_flops
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.model import GPT, Block, CausalSelfAttention, Config, LLaMAMLP
from lit_gpt.packed_dataset import CombinedDataset
from lit_gpt.utils import CycleIterator, chunked_cross_entropy, num_parameters

# import thunder

# ---------------
# System settings
# ---------------

# The model name determines the architecture of the model
# See lit_gpt/config.py for a list of supported models
# For example, set Llama-2-7b-hf
model_name = "tiny-llama-1.1b"
# model_name = "pythia-14m"

# Name of the folder where logs and checkpoints will be saved
# name = "lit-tiny-llama-1.1b-v1"
name = "lit-tiny-llama-162M-v3"
out_dir = Path(os.getenv("LIGHTNING_ARTIFACTS_DIR", "out")) / name

# Choose 'tensorboard', 'wandb', or 'csv'
logger_name = "wandb"

# Training will require all GPUs. Not recommended to run on CPU.
devices = torch.cuda.device_count() or 1

# ---------------
# Hyperparameters
# ---------------

# The learning rate, try to tune it if you deviate a lot from the TinyLlama architecture,
# sequence length, batch size, etc. See also 'warmup_steps' below
learning_rate = 6e-4

# The total batch size within a single machine
global_batch_size = 512

# Set micro batch size as high as possible to efficiently use your GPU memory and push GPUs to 99%+ utilization
# Lower it if you are short on memory
micro_batch_size = 32

# Stop training after this many tokens have been trained on (across all GPUs)
# max_tokens = int(3e12)  # 3 trillion
# max_tokens = int(53e6*5)  # num training example * len context window --> azwiki
max_tokens = int(3e+9*3) # 9M samples, 3B for 3 epochs
# The number of (optimizer) steps until reaching the max learning rate
# After that, the learning rate will decay with cosine (see scheduler below)
warmup_steps = 500 # 146484 * 0.01 = ~1400

# Logging, evaluation, and checkpointing intervals
log_step_interval = 1
eval_iters = 100
save_step_interval = 100
eval_step_interval = 100

# Optimizer hyperparams, leave default
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
min_lr = 6e-5

batch_size = global_batch_size // devices
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
warmup_iters = warmup_steps * gradient_accumulation_steps
log_iter_interval = log_step_interval * gradient_accumulation_steps


hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}


def setup(resume: Union[bool, Path] = False, context_size: int = 2048):
    logger = choose_logger(logger_name, name=name, resume=resume)

    strategy = FSDPStrategy(
        # Auto-wrap policy determines which parts of the model will get sharded
        # Here we want to shard the large transformer blocks
        auto_wrap_policy={Block}, 
        
        # 'full' will save a regular checkpoint
        # 'sharded' will save a distributed checkpoint, use if training models > 1B
        state_dict_type="full", 
        
        # Sharding strategy determines how your model gets sharded across GPUs and machines
        # HYBRID_SHARD: Use for models that fit into a single machine, e.g. TinyLlama 1B - 3B on an 8xA100
        #               This will shard the model within the machine, but replicate across machines
        #               Useful to avoid slow network bottleneck between machines
        #               
        # FULL_SHARD:   Use for very large models that don't fit into a machine. The model gets sharded
        #               across all GPUs on all machines. Requires a cluster with fast inter-node network (like on Lightning AI)
        #               You typically need this with models above 3B parameters.
        sharding_strategy="HYBRID_SHARD",

        # Enable activation checkpointing to save memory, if needed
        # Training will be slower if this is enabled
        # activation_checkpointing_policy={Block},
    )
    
    fabric = L.Fabric(devices=devices, strategy=strategy, precision="bf16-true", loggers=[logger])
    fabric.launch()

    fabric.print(hparams)
    if logger_name in ("tensorboard", "wandb"):
        fabric.logger.log_hyperparams(hparams)

    main(fabric, resume, context_size)


def main(fabric, resume, context_size):
    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(
        # Choose a name here from a selection of configs of popular models available in Lit-GPT
        # See lit_gpt/config.py for a full list
        # For example, set 'Llama-2-7b-hf' if you want a larger one than TinyLlama
        model_name, 
        # The block size, or sequence length, or context size determines how long the sequences
        # are we use to train the model. The bigger this is, the more memory training will require!
        # Reduce this size if you run out of memory.
        block_size=context_size,
        vocab_size=32000,

        # Alternatively, you can select the internal sizes yourself to make a custom-sized model.
        # There are 4 impactful sizes to choose here. For more details, see lit_gpt/model.py.
        
        # 162M
        n_layer=12,
        n_head=12,
        n_embd=768,
        intermediate_size=3072,

        # 1.5B
        # n_layer=32
        # n_head=32,
        # n_embd=2048,

        # 2B
        # n_layer=32,
        # n_head=32,
        # n_embd=2560,

        # 3B
        # n_layer=40,
        # n_head=32,
        # n_embd=2560,
        # intermediate_size=6912,
    )

    train_dataloader, val_dataloader = create_dataloaders(batch_size=micro_batch_size, block_size=config.block_size)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = GPT(config)
        model.apply(partial(init_weights, n_layer=config.n_layer, n_embd=config.n_embd))
        # model = thunder.jit(model)

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters: {num_parameters(model):,}")

    model = torch.compile(model)
    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
    )
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    if resume is True:
        resume = max(out_dir.glob("*.pth"), key=(lambda p: int(p.name.split("-")[1])))
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, resume)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(fabric, state, train_dataloader, val_dataloader, resume):
    model = state["model"]
    optimizer = state["optimizer"]

    validate(fabric, model, val_dataloader, max_iters=2)  # sanity check
    throughput = ThroughputMonitor(fabric, window_size=5)

    with torch.device("meta"):
        meta_model = GPT(model.config)
        x = torch.randint(0, 1, (micro_batch_size, meta_model.config.block_size))
        model_fwd = lambda: meta_model(x)
        model_loss = lambda y: chunked_cross_entropy(y, x, chunk_size=0)
        measured_flops = measure_flops(meta_model, model_fwd, model_loss)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    max_tokens_per_device = max_tokens // fabric.world_size
    tokens_per_iter = micro_batch_size * model.config.block_size
    max_iters = max_tokens_per_device // tokens_per_iter
    initial_iter = state["iter_num"]
    train_iterator = CycleIterator(train_dataloader)
    print(f"Number of training iters: {max_iters}")

    # resume data loader state by fast-forwarding through all seen batches
    # drop this once streaming dataset supports proper resuming
    if resume:
        resume_t0 = time.perf_counter()
        for resume_iter in range(initial_iter):
            next(train_iterator)
            if resume_iter % 1000 == 0:
                fabric.print(f"Resuming dataset: {resume_iter} / {initial_iter}")
        fabric.barrier()
        fabric.print(
            f"Resuming data loader finished. Took {time.perf_counter() - resume_t0:.1f} seconds to reach iteration"
            f" {initial_iter}, epoch {train_iterator.epoch}."
        )

    running_loss = RunningMean(window=gradient_accumulation_steps, sync_on_compute=False).to(fabric.device)
    fabric.barrier()
    total_t0 = time.perf_counter()

    for train_data in train_iterator:
        if state["iter_num"] >= max_iters:
            print(f"Current ite {state['iter_num']}")
            break

        # determine and set the learning rate for this iteration
        lr = get_lr(state["iter_num"], max_iters) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        state["iter_num"] += 1
        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0 : model.config.block_size].contiguous().long()
        targets = train_data[:, 1 : (model.config.block_size + 1)].contiguous().long()

        is_accumulating = state["iter_num"] % gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = chunked_cross_entropy(logits, targets)
            fabric.backward(loss / gradient_accumulation_steps)

        running_loss.update(loss.detach())

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1

        if state["iter_num"] % log_iter_interval == 0:
            loss = running_loss.compute().item()  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            throughput.update(
                time=(t1 - total_t0),
                flops=(measured_flops * log_iter_interval),
                batches=state["iter_num"],
                samples=(state["iter_num"] * micro_batch_size),
                lengths=(state["iter_num"] * micro_batch_size * model.config.block_size),
            )
            metrics = {
                "loss": loss,
                "iter": state["iter_num"],
                "step": state["step_count"],
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "remaining_time": (
                    (t1 - total_t0) / (state["iter_num"] - initial_iter) * (max_iters - state["iter_num"])
                ),
                "tokens": state["iter_num"] * micro_batch_size * model.config.block_size,
                "total_tokens": state["iter_num"] * micro_batch_size * model.config.block_size * fabric.world_size,
                "learning_rate": lr,
            }

            fabric.print(
                f"iter {metrics['iter']} | step {metrics['step']}: loss {metrics['loss']:.4f}, iter time:"
                f" {metrics['iter_time'] * 1000:.2f} ms{' (optimizer.step),' if not is_accumulating else ','}"
                f" remaining time: {metrics['remaining_time'] / 3600 / 24:.2f} days"
                f" {metrics['remaining_time'] / 3600:.4f} hours"
            )

            throughput_metrics = throughput.compute()
            metrics.update(throughput_metrics)
            fabric.log_dict(metrics, step=state["iter_num"])

        if val_dataloader is not None and not is_accumulating and state["step_count"] % eval_step_interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader, max_iters=eval_iters)
            val_loss = val_loss.item()
            td = time.perf_counter() - t0

            fabric.print(f"iter {state['iter_num']}: val loss {val_loss:.4f}, val time: {td * 1000:.2f} ms")
            metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
            fabric.log_dict(metrics, step=state["iter_num"])
            fabric.barrier()

        if not is_accumulating and state["step_count"] % save_step_interval == 0:
            checkpoint_path = out_dir / f"step-{state['step_count']:08d}.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)


@torch.no_grad()
def validate(fabric: L.Fabric, model: nn.Module, val_dataloader: DataLoader, max_iters: int) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(max_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= max_iters:
            break
        input_ids = val_data[:, 0 : model.config.block_size].contiguous().long()
        targets = val_data[:, 1 : (model.config.block_size + 1)].contiguous().long()
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets)
        losses[k] = loss

    model.train()
    return losses.mean()


def create_dataloaders(batch_size: int, block_size: int) -> Tuple[DataLoader, DataLoader]:
    from lightning.data import StreamingDataset
    from lightning.data.streaming.item_loader import TokensLoader

    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1

    # train_datasets = [
    #     StreamingDataset(
    #         input_dir="/teamspace/s3_connections/tinyllama-template/slimpajama/train",
    #         item_loader=TokensLoader(block_size=effective_block_size),
    #         shuffle=True,
    #         drop_last=True,
    #     ),
    #     StreamingDataset(
    #         input_dir="/teamspace/s3_connections/tinyllama-template/starcoder",
    #         item_loader=TokensLoader(block_size=effective_block_size),
    #         shuffle=True,
    #         drop_last=True,
    #     ),
    # ]

    # Mix SlimPajama data and Starcoder data with these proportions:
    # weights = (0.693584, 0.306416)
    # combined_dataset = CombinedDataset(datasets=train_datasets, seed=42, weights=weights)

    # train_dataloader = DataLoader(
    #     combined_dataset, batch_size=batch_size, pin_memory=True, num_workers=8, drop_last=True
    # )

    # val_dataset = StreamingDataset(
    #     input_dir="/teamspace/s3_connections/tinyllama-template/slimpajama/val",
    #     item_loader=TokensLoader(block_size=eff-rangeective_block_size),
    #     shuffle=True,
    #     # Consider setting to False, but we would lose some samples due to truncation when world size > 1
    #     drop_last=True,
    # )

    train_dataset = StreamingDataset(
            input_dir="/teamspace/studios/this_studio/data/simhash-dedup-train",
            item_loader=TokensLoader(block_size=effective_block_size),
            shuffle=True,
            drop_last=True,
        )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, pin_memory=True, num_workers=8, drop_last=True
    )

    val_dataset = StreamingDataset(
        input_dir="/teamspace/studios/this_studio/data/simhash-dedup-val",
        item_loader=TokensLoader(block_size=effective_block_size),
        shuffle=True,
        # Consider setting to False, but we would lose some samples due to truncation when world size > 1
        drop_last=True,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=8, drop_last=True)
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with warmup)
def get_lr(it: int, lr_decay_iters: int) -> int:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def init_weights(module: nn.Module, n_layer: int, n_embd: int):
    # Follows GPT-NeoX: https://arxiv.org/abs/2204.06745
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / n_embd))
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / n_embd))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    for name, param in module.named_parameters():
        if name == "proj.weight" and isinstance(module, (LLaMAMLP, CausalSelfAttention)):
            nn.init.normal_(param, mean=0.0, std=(1 / math.sqrt(n_embd) / n_layer))


def choose_logger(logger_name: str, name: str, resume: Union[bool, Path], *args, **kwargs):
    if logger_name == "csv":
        return CSVLogger(root_dir=(out_dir / "logs"), name="csv", *args, **kwargs)
    if logger_name == "tensorboard":
        return TensorBoardLogger(root_dir=(out_dir / "logs"), name="tensorboard", *args, **kwargs)
    if logger_name == "wandb":
        return WandbLogger(project="tinyllama", name=name, resume=(resume is not False), *args, **kwargs)
    raise ValueError(f"`logger={logger_name}` is not a valid option.")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI
    from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_2

    if not _TORCH_GREATER_EQUAL_2_2:
        raise ImportError("The tinyllama.py training script requires PyTorch 2.2 (nightly) or higher to run.")

    CLI(setup)
