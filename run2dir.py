import argparse
import logging
import sys
import os
import time
import pathlib

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger("TfPoseEstimatorRun")
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tf-pose-estimation run")
    parser.add_argument("--dir", type=str, default="./images/")
    parser.add_argument(
        "--model",
        type=str,
        default="cmu",
        help="cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small",
    )
    parser.add_argument(
        "--resize",
        type=str,
        default="0x0",
        help="if provided, resize images before they are processed. "
        "default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ",
    )
    parser.add_argument(
        "--resize-out-ratio",
        type=float,
        default=4.0,
        help="if provided, resize heatmaps before they are post-processed. default=4.0",
    )

    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    image_dir = pathlib.Path(args.dir)
    image_list = image_dir.glob("*.png")
    # now sort it
    def sort_images(pp):
        stem1 = pp.stem  # get rid of .png
        num_str = os.path.splitext(stem1)[1]
        return int(num_str[1:])

    image_list = sorted(image_list, key=sort_images)
    xys = np.zeros((18, 2, len(image_list)))
    # estimate human poses from a directory of images

    for tt, image_file_name in enumerate(image_list):
        image = common.read_imgfile(str(image_file_name), None, None)
        if image is None:
            logger.error("Image can not be read, path=%s" % args.image)
            sys.exit(-1)

        t = time.time()
        humans = e.inference(
            image,
            resize_to_default=(w > 0 and h > 0),
            upsample_size=args.resize_out_ratio,
        )
        print(f"len(humans): {len(humans)}, image.shape: {image.shape}, {image.dtype}")
        # len(humanslen(humans): 1, image.shape: (480, 582, 3), uint8
        try:
            human0 = humans[0]
            for bp in human0.body_parts:
                xys[bp, 0, tt] = human0.body_parts[bp].x
                xys[bp, 1, tt] = human0.body_parts[bp].y
        except:
            pass
        # humans holds a list of objects class Human
        # this has a list of body parts in if let a = human.body_parts[0]
        # body parts have a (a.x, a.y) and a score a.score which relates to confidence
        # the name of the body parts are given in the class enum. The coordinate system
        # has the y values increasing downwards from the upper left corner of 0,0
        #
        # class CocoPart(Enum):
        # Nose = 0
        # Neck = 1
        # RShoulder = 2
        # RElbow = 3
        # RWrist = 4
        # LShoulder = 5
        # LElbow = 6
        # LWrist = 7
        # RHip = 8
        # RKnee = 9
        # RAnkle = 10
        # LHip = 11
        # LKnee = 12
        # LAnkle = 13
        # REye = 14
        # LEye = 15
        # REar = 16
        # LEar = 17
        # Background = 18

        elapsed = time.time() - t

        logger.info("inference image: %s in %.4f seconds." % (image_file_name, elapsed))

        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        try:

            # import matplotlib.pyplot as plt

            # fig = plt.figure()
            # a = fig.add_subplot(2, 2, 1)
            # a.set_title('Result')
            # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
            # bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

            # # show network output
            # a = fig.add_subplot(2, 2, 2)
            # plt.imshow(bgimg, alpha=0.5)
            # tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
            # plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
            # plt.colorbar()

            # tmp2 = e.pafMat.transpose((2, 0, 1))
            # tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
            # tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

            # a = fig.add_subplot(2, 2, 3)
            # a.set_title('Vectormap-x')
            # # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
            # plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
            # plt.colorbar()

            # a = fig.add_subplot(2, 2, 4)
            # a.set_title('Vectormap-y')
            # # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
            # plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
            # plt.colorbar()
            # plt.show()
            pass
        except Exception as e:
            logger.warning("matplitlib error, %s" % e)
            cv2.imshow("result", image)
            cv2.waitKey()
    # cv2.waitKey()
    np.save("xys.npy", xys)

