from pydantic import BaseModel


class OpenWorldVideoLogSingleton(BaseModel):
    video_log_paths: list[str]
    video_id: str

    def __hash__(self):
        sorted_paths = sorted(self.video_log_paths)
        return hash("".join(sorted_paths) + self.video_id)


class OpenWorldVideoLog(BaseModel):
    video_logs: list[OpenWorldVideoLogSingleton]

    def __hash__(self):
        return sum(hash(video) for video in self.video_logs)

    def __len__(self):
        return len(self.video_logs)
