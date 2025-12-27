import re

from pydantic import BaseModel

from data.s3.s3_utils import S3Manager


class S3Frame(BaseModel):
    frame_num: int
    obj_key: str


class S3FrameList(BaseModel):
    directory: str
    frame_list: list[S3Frame]


def list_frames_in_s3(s3_manager: S3Manager, source_dir: str) -> list[S3FrameList]:
    frame_objects = s3_manager.list_objects(prefix=source_dir, suffix=".png")

    # Group by directory (game/episode)
    directories: list[S3FrameList] = []
    directory_mapping: dict[str, list[S3Frame]] = {}
    for obj_key in frame_objects:
        if "frame_" in obj_key:
            # Extract directory path
            dir_path = "/".join(obj_key.split("/")[:-1])
            if dir_path not in directory_mapping:
                directory_mapping[dir_path] = []

            # Extract frame number
            filename = obj_key.split("/")[-1]
            match = re.search(r"frame_(\d+)", filename)
            if match:
                frame_num = int(match.group(1))
                directory_mapping[dir_path].append(
                    S3Frame(frame_num=frame_num, obj_key=obj_key)
                )

    for dir_path, frame_list in directory_mapping.items():
        directories.append(S3FrameList(directory=dir_path, frame_list=frame_list))

    return directories
