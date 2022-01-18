import numpy as np
from dataclasses import dataclass
from src.utils.python import get_full_class_reference


class MiddleFramesRemover:
    def __init__(self, past_context=10, future_context=1, middle_frames=5):
        self.past_context = past_context
        self.future_context = future_context
        self.middle_frames = middle_frames

    def __call__(self, frames):
        assert self.past_context + self.future_context + self.middle_frames < len(
            frames), "Sequence is to short {0:} for the past context {1:}, future context {2:}, middle frames {3:}.".format(
            len(frames), self.past_context, self.future_context, self.middle_frames)

        past_plus_middle = self.past_context + self.middle_frames

        past_context_frames = frames[:self.past_context]
        future_context_frames = frames[past_plus_middle:past_plus_middle + self.future_context]
        missing_frames = frames[self.past_context:past_plus_middle]
        return past_context_frames, future_context_frames, missing_frames


@dataclass
class MiddleFramesRemoverOptions:
    _target_: str = get_full_class_reference(MiddleFramesRemover)
    past_context: int = 10
    future_context: int = 1
    middle_frames: int = 5


class RandomMiddleFramesRemover:
    def __init__(self, min_past_context=1,
                 max_past_context=15,
                 max_future_context=5,
                 min_middle_frames=5,
                 max_middle_frames=40,
                 weighted_middle_frames=True):
        self.max_past_context = max_past_context
        self.min_past_context = min_past_context
        self.max_future_context = max_future_context
        self.min_middle_frames = min_middle_frames
        self.max_middle_frames = max_middle_frames
        self.weighted_middle_frames = weighted_middle_frames


        self.middle_frame_lengths = np.array(range(self.min_middle_frames, self.max_middle_frames + 1))
        self.middle_frame_weights = 1.0 / self.middle_frame_lengths
        self.middle_frame_weights = self.middle_frame_weights / sum(self.middle_frame_weights)

    def __call__(self, frames):
        assert self.min_past_context > 0, "min_past_context must be positive"
        assert self.max_past_context + self.max_middle_frames + 1 <= len(
            frames), f"Sequence is to short for the max_past_context context {self.max_past_context}"
        past_context = np.random.randint(self.min_past_context, self.max_past_context + 1)
        remaining_frames = len(frames) - past_context
        if self.weighted_middle_frames:
            # Here we sample shorter windows more often, because the number of short windows in the dataset is larger
            middle_frame_idx_min = min(self.min_middle_frames, remaining_frames - 1) - self.min_middle_frames
            middle_frame_idx_max = min(self.max_middle_frames + 1, remaining_frames) - self.min_middle_frames
            middle_frames = np.random.choice(self.middle_frame_lengths[middle_frame_idx_min:middle_frame_idx_max],
                                             replace=False,
                                             p=self.middle_frame_weights[middle_frame_idx_min:middle_frame_idx_max])
        else:
            middle_frames = np.random.randint(min(self.min_middle_frames, remaining_frames - 1),
                                              min(self.max_middle_frames + 1, remaining_frames))
        past_plus_middle = past_context + middle_frames
        remaining_frames = len(frames) - past_plus_middle
        future_context = np.random.randint(1, min(self.max_future_context + 1, remaining_frames + 1))
        remaining_frames = len(frames) - (past_context + middle_frames + future_context)
        start_frame = np.random.randint(0, remaining_frames + 1)
        past_context += start_frame
        past_plus_middle += start_frame
        past_context_frames = frames[start_frame:past_context]
        future_context_frames = frames[past_plus_middle:past_plus_middle + future_context]
        missing_frames = frames[past_context:past_plus_middle]
        return past_context_frames, future_context_frames, missing_frames


@dataclass
class RandomMiddleFramesRemoverOptions:
    _target_: str = get_full_class_reference(RandomMiddleFramesRemover)
    min_past_context: int = 10
    max_past_context: int = 10
    max_future_context: int = 1
    min_middle_frames: int = 5
    max_middle_frames: int = 5
    weighted_middle_frames: bool = True

