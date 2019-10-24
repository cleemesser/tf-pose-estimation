# note can also use imutils to read in another thread
import argparse
import logging
import time
import sys
import pathlib


import matplotlib.pyplot as plt
import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from tf_pose.estimator import Human, BodyPart

from skeleton_serialize import Skeleton

logger = logging.getLogger("TfPoseEstimator-Video")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

NONETYPE = type(None)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="tf-pose-estimation Video")
    parser.add_argument("--show", "-s", type=bool, default=False)
    parser.add_argument("--video", type=str, default="")
    parser.add_argument(
        "--resolution",
        type=str,
        default="432x368",
        help="network input resolution. default=432x368",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cmu",
        help="cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small",
    )
    parser.add_argument(
        "--show-process",
        type=bool,
        default=False,
        help="for debug purpose, if enabled, speed for inference is dropped.",
    )
    parser.add_argument(
        "--showBG", type=bool, default=False, help="False to show skeleton only."
    )
    parser.add_argument(
        "--resize-out-ratio",
        type=float,
        default=4.0,
        help="if provided, resize heatmaps before they are post-processed. default=4.0",
    )
    args = parser.parse_args()
    print(args)

    logger.debug("initialization %s : %s" % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    # print(f"w,h: {(w,h)}")
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    cap = cv2.VideoCapture(args.video)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skeletons = Skeleton(args.video, frame_rate=frame_rate, num_frames=num_frames)
    if cap.isOpened() is False:
        print("Error opening video stream or file")

    # something wrong here with either/both color space and float vs uint8
    # pltimage = None
    while cap.isOpened():
        ret_val, image = cap.read()

        # try resizing image to fit network?
        if type(image) == np.ndarray:
            image_use = np.copy(image)

            humans = e.inference(
                image_use,
                resize_to_default=(w > 0 and h > 0),
                upsample_size=args.resize_out_ratio,
            )
            # print(humans, w, h)  # here is where we will output the pose information
            skeletons.append(humans)
            if args.show:
                if not args.showBG:
                    image_use = np.zeros(image_use.shape)
                image_use = TfPoseEstimator.draw_humans(
                    image_use, humans, imgcopy=False
                )

                cv2.putText(
                    image_use,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("tf-pose-estimation result", image_use)
                fps_time = time.time()
        elif type(image) == NONETYPE:
            # got to end of video file
            break
        if cv2.waitKey(1) == 27:
            break
    # done going through video
    cv2.destroyAllWindows()
    skeletons.cleanup()
logger.debug("finished+")
