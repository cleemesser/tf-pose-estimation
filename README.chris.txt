built tf_pose/pa_process/
swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
---
update: opencv 3.4 from the default channel did not work
so set chanel_priority to flexible and then did
conda install opencv -c conda-forge -> which lead to a big exchange and opencv 4.1.1 which worked

Mostly successful try: tf14py37

conda create -n tf14py37 python=3.7 tensorflow-gpu cython opencv tqdm 
then did 
conda install pip  # to update it
pip install slidingwindow

pip install -e .
the basic run partially worked, but there is some warning

I got among other errors and warnings:
```
WARNING:tensorflow:From /mnt/home2/clee/code/tf-pose-estimation/tf_pose/estimator.py:341: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

2019-10-12 12:52:54,983 WARNING From /mnt/home2/clee/code/tf-pose-estimation/tf_pose/estimator.py:341: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

WARNING:tensorflow:From /mnt/home2/clee/code/tf-pose-estimation/tf_pose/estimator.py:342: The name tf.image.resize_area is deprecated. Please use tf.compat.v1.image.resize_area instead.

2019-10-12 12:52:54,984 WARNING From /mnt/home2/clee/code/tf-pose-estimation/tf_pose/estimator.py:342: The name tf.image.resize_area is deprecated. Please use tf.compat.v1.image.resize_area instead.

WARNING:tensorflow:From /mnt/home2/clee/code/tf-pose-estimation/tf_pose/tensblur/smoother.py:96: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.

2019-10-12 12:52:54,991 WARNING From /mnt/home2/clee/code/tf-pose-estimation/tf_pose/tensblur/smoother.py:96: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.

WARNING:tensorflow:From /mnt/home2/clee/code/tf-pose-estimation/tf_pose/estimator.py:354: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2019-10-12 12:52:54,999 WARNING From /mnt/home2/clee/code/tf-pose-estimation/tf_pose/estimator.py:354: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2019-10-12 12:52:55.057693: W tensorflow/compiler/jit/mark_for_compilation_pass.cc:1412] (One-time warning): Not using XLA:CPU for cluster because envvar TF_XLA_FLAGS=--tf_xla_cpu_global_jit was not set.  If you want XLA:CPU, either set that envvar, or use experimental_jit_scope to enable XLA:CPU.  To confirm that XLA is active, pass --vmodule=xla_compilation_cache=1 (as a proper command-line flag, not via TF_XLA_FLAGS) or set the envvar XLA_FLAGS=--xla_hlo_profile.
2019-10-12 12:52:55.972436: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudnn.so.7
[2019-10-12 12:52:56,939] [TfPoseEstimatorRun] [INFO] inference image: ./images/p1.jpg in 0.0231 seconds.
2019-10-12 12:52:56,939 INFO inference image: ./images/p1.jpg in 0.0231 seconds.
(tf14py37) clee@lotus:~/code/tf-pose-estimation$ vim README.chris.txt 
```


Failed tries:
pip install tensorflow-gpu==1.4.1

this seems to use cuda8

pip install tensorflow-gpu==1.6
this seems to require cuda 9.0 which I do not have installed


since I have cuda 9.1, I will try:
https://github.com/evdcush/TensorFlow-wheels/releases/download/tf-1.10.0-gpu-9.1-mkl/tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl
tensorflow 1.10

