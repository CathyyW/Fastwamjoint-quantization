"""Opt-in bounded pinned-memory saved-tensor offload for sensitivity backward.

Copies use the current compute stream: later in-place writes cannot race the
snapshot. This removes host blocking for pinned copies, not the GPU copy work.
No cross-stream prefetch or unlimited pinning is claimed.
"""
from contextlib import contextmanager

import torch


class _PinnedTensor:
    def __init__(self, host, event, device, owner, size):
        self.host, self.event, self.device = host, event, device
        self.owner, self.size = owner, size

    def __del__(self):
        self.owner.active_bytes -= self.size


class BoundedSavedTensorOffload:
    def __init__(self, model, *, max_pinned_bytes=2 << 30, min_bytes=1 << 20):
        if max_pinned_bytes < 0 or min_bytes < 0:
            raise ValueError("Offload byte limits must be nonnegative.")
        self.parameter_storages = {p.untyped_storage().data_ptr() for p in model.parameters()
                                   if p.device.type == 'cuda'}
        self.max_pinned_bytes, self.min_bytes = max_pinned_bytes, min_bytes
        self.active_bytes = 0
        self.peak_bytes = 0
        self.pinned_copies = 0
        self.fallback_copies = 0

    def pack(self, tensor):
        size = tensor.numel() * tensor.element_size()
        if (tensor.device.type != 'cuda' or size < self.min_bytes
                or tensor.untyped_storage().data_ptr() in self.parameter_storages
                or (tensor.ndim >= 2 and tensor.shape[-1] % 4 != 0)):
            return tensor
        if self.active_bytes + size > self.max_pinned_bytes:
            self.fallback_copies += 1
            return (tensor.device, tensor.to('cpu', non_blocking=False))
        # Allocation failures are not suppressed: an unexpected resource error
        # must not turn into silent numerics or an unbounded retry loop.
        host = torch.empty_like(tensor, device='cpu', pin_memory=True,
                                memory_format=torch.preserve_format)
        stream = torch.cuda.current_stream(tensor.device)
        host.copy_(tensor, non_blocking=True)
        tensor.record_stream(stream)
        event = torch.cuda.Event()
        event.record(stream)
        self.active_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.active_bytes)
        self.pinned_copies += 1
        return _PinnedTensor(host, event, tensor.device, self, size)

    def unpack(self, payload):
        if isinstance(payload, torch.Tensor):
            return payload
        if isinstance(payload, tuple):
            device, tensor = payload
            return tensor.to(device, non_blocking=False)
        # A backward may run on a different stream. Enqueue the dependency;
        # never read the pinned CPU payload on the host before this completes.
        torch.cuda.current_stream(payload.device).wait_event(payload.event)
        return payload.host.to(payload.device, non_blocking=True)

    @contextmanager
    def context(self):
        with torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack):
            yield self
