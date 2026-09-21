from __future__ import annotations

from dataclasses import dataclass

import torch

from .topology import LinearSite


@dataclass(frozen=True)
class FastWAMStreamConfig:
    """Architecture-native semantic streams used by local gamma gains.

    LIBERO's usual 9 input frames become 3 VAE latent frames, while 32 action
    tokens are split into two chunks aligned with the two future latent steps.
    """

    video_latent_frames: int = 3
    action_chunks: int = 2
    proprio_tokens: int = 1
    split_context: bool = True

    def validate(self) -> None:
        if self.video_latent_frames <= 0 or self.action_chunks <= 0:
            raise ValueError("video_latent_frames and action_chunks must be positive.")
        if self.proprio_tokens < 0:
            raise ValueError("proprio_tokens must be non-negative.")


@dataclass(frozen=True)
class ResolvedStreamLayout:
    names: tuple[str, ...]
    token_counts: tuple[int, ...]

    def validate(self, sequence_length: int) -> None:
        if not self.names or len(self.names) != len(self.token_counts):
            raise ValueError("A stream layout needs one non-empty name per token count.")
        if any(count <= 0 for count in self.token_counts):
            raise ValueError("Every stream must contain at least one token.")
        if sum(self.token_counts) != int(sequence_length):
            raise ValueError("Stream token counts do not cover the input sequence.")

    @property
    def num_streams(self) -> int:
        return len(self.names)

    def slices(self) -> tuple[slice, ...]:
        result: list[slice] = []
        start = 0
        for count in self.token_counts:
            result.append(slice(start, start + count))
            start += count
        return tuple(result)


def _equal_groups(prefix: str, sequence_length: int, groups: int) -> ResolvedStreamLayout:
    if sequence_length % groups:
        raise ValueError(
            f"Sequence length {sequence_length} is not divisible by {groups} {prefix} streams."
        )
    count = sequence_length // groups
    return ResolvedStreamLayout(
        names=tuple(f"{prefix}_{index}" for index in range(groups)),
        token_counts=(count,) * groups,
    )


def resolve_stream_layout(
    site: LinearSite,
    sequence_length: int,
    config: FastWAMStreamConfig,
) -> ResolvedStreamLayout:
    config.validate()
    sequence_length = int(sequence_length)
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")

    if site.stream_kind == "context":
        if not config.split_context or config.proprio_tokens == 0:
            layout = ResolvedStreamLayout(("context",), (sequence_length,))
        else:
            text_tokens = sequence_length - config.proprio_tokens
            if text_tokens <= 0:
                raise ValueError("Context must contain text tokens before appended proprio tokens.")
            layout = ResolvedStreamLayout(
                ("text", "proprio"),
                (text_tokens, config.proprio_tokens),
            )
    elif site.expert == "video":
        layout = _equal_groups("latent_frame", sequence_length, config.video_latent_frames)
    else:
        layout = _equal_groups("action_chunk", sequence_length, config.action_chunks)
    layout.validate(sequence_length)
    return layout


def rows_for_stream(x: torch.Tensor, stream_slice: slice) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("Linear input must have a token and hidden dimension.")
    selected = x[..., stream_slice, :]
    return selected.reshape(-1, selected.shape[-1])


def expand_stream_gains(
    x: torch.Tensor,
    gains: torch.Tensor,
    layout: ResolvedStreamLayout,
) -> torch.Tensor:
    if x.ndim < 2 or x.shape[-2] != sum(layout.token_counts):
        raise ValueError("Runtime tensor does not match its stream layout.")
    gains = torch.as_tensor(gains, device=x.device, dtype=x.dtype)
    if gains.shape != (layout.num_streams,):
        raise ValueError(f"Expected {layout.num_streams} gains, got {tuple(gains.shape)}.")
    rows = torch.repeat_interleave(
        gains,
        torch.tensor(layout.token_counts, device=x.device),
    )
    shape = [1] * x.ndim
    shape[-2] = x.shape[-2]
    return rows.reshape(shape)
