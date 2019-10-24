# some starting code to serialize the output of this openpose based
# estimator when run again a video

import numpy as np

# import cv2  # get rid of this opencv dependency!!! just pass in the info

import pathlib
from tf_pose.estimator import Human, BodyPart
import json


class SkeletonJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Human):
            return obj.body_parts
        if isinstance(obj, BodyPart):
            return {obj.part_idx: (obj.x, obj.y)}
        return json.JSONEncoder.default(self, obj)


class Skeleton:
    def __init__(self, video_name, frame_rate, num_frames):
        """
        for the momement will accept the opencv video capture @cap
        and the video file name
        """
        file_path = pathlib.Path(video_name)
        frame_rate = frame_rate  # e.g. cap.get(cv2.CAP_PROP_FPS)
        num_frames = num_frames  # e.g. int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = frame_rate
        self.frame_num = 0
        self.num_humans = np.zeros(num_frames, dtype=np.uint32)
        self.frame2poses = {}
        self.file_name_base = str(file_path.parent / file_path.stem)
        self.file_name_json = self.file_name_base + ".json"
        self.fp = None

    def append(self, humans):  # this writes to disk instead of storing in memory
        """create json files to represent the poses for each frame on each line"""
        if not self.fp:
            self.fp = open(self.file_name_json, "w+")  # overwrites is that best???
        num_humans = len(humans)
        self.num_humans[self.frame_num] = num_humans
        #  consider skipping writing ifno about frames without any humans
        self.fp.write(json.dumps(humans, cls=SkeletonJsonEncoder))
        self.fp.write("\n")
        self.frame_num += 1

    def cleanup(self):
        self.fp.close()
        np.save(self.file_name_base + "num_humans.npy", self.num_humans)

    def read(self, num_human_file_name, json_file_name):
        """read an already created file and make sense of it"""
        self.num_humans = np.load(num_human_file_name)
        nh = self.num_humans
        nonzeroindices = np.argwhere(
            nh > 0
        ).flatten()  # the indices where there are detected humans
        # these are the frame number numbers to act as keys to the dictionary we are going to make
        frame2poses = {}
        line_num = 0
        print(f"about to open {json_file_name}")
        with open(json_file_name) as fp:
            print("reading file")
            while True:
                line = fp.readline()
                # print(line)
                # speed up, if nh[line] == 0, then can skip or just = []
                if line:
                    frame2poses[line_num] = json.loads(line)
                    line_num += 1
                else:
                    print("breaking")
                    break
        self.frame2poses = frame2poses


def test_read():
    import cv2

    video = "/home/cleemess/StanfordOpenPoseAnalysisVideo/myoclonic-atonic-vid02.mp4"
    cap = cv2.VideoCapture(video)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    # 29.97002997002997
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # In [15]: num_frames
    # Out[17]: 389
    sk = Skeleton(video, frame_rate, num_frames)
    hfn = "/home/cleemess/StanfordOpenPoseAnalysisVideo/myoclonic-atonic-vid02num_humans.npy"
    jfn = "/home/cleemess/StanfordOpenPoseAnalysisVideo/myoclonic-atonic-vid02.json"
    sk.read(hfn, jfn)
    return sk


if __name__ == "__main__":
    sk = test_read()
