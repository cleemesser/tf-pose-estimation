# IPython log file

import cv2
get_ipython().run_line_magic('pinfo', 'cv2.imread')
# get the first image
im0 = cv2.imread('/mnt/home2/clee/code/eegml/seizure-videos/myoclonic-atonic-vid02-dir/myoclonic-atonic-vid02.1.png, cv2.IMREAD_COLOR)
im0 = cv2.imread('/mnt/home2/clee/code/eegml/seizure-videos/myoclonic-atonic-vid02-dir/myoclonic-atonic-vid02.1.png', cv2.IMREAD_COLOR)
im0
get_ipython().run_line_magic('pinfo', 'cv2.VideoCapture')
cap = cv2.VideoCapture('/mnt/home2/clee/code/eegml/seizure-videos/myoclonic-atonic-vid02.mp4')
get_ipython().set_next_input('ret, vim0 = cap.read');get_ipython().run_line_magic('pinfo', 'cap.read')
ret, vim0 = cap.read()
vim0.shape
im0.shape
dim0 = im0 - vim0
dim0.max()
dim0.min()
get_ipython().run_line_magic('logstart', '')
get_ipython().run_line_magic('run', 'run_video2.py --video ~/code/eegml/seizure-videos/myoclonic-atonic-vid02.mp4')
get_ipython().run_line_magic('run', 'run_video2.py --video ~/code/eegml/seizure-videos/myoclonic-atonic-vid02.mp4')
get_ipython().run_line_magic('run', 'run_video2.py --showBG True --video ~/code/eegml/seizure-videos/myoclonic-atonic-vid02.mp4')
get_ipython().run_line_magic('run', 'run_video2.py --showBG True --video ~/code/eegml/seizure-videos/myoclonic-atonic-vid02.mp4')
get_ipython().run_line_magic('run', 'run_video2.py --showBG True --video ~/code/eegml/seizure-videos/myoclonic-atonic-vid02.mp4')
get_ipython().run_line_magic('run', 'run_video2.py --showBG True --video ~/code/eegml/seizure-videos/myoclonic-atonic-vid02.mp4')
get_ipython().run_line_magic('run', 'rund2dir --dir ~/code/eegml/seizure-videos/myoclonic-atonic-vid02-dir')
get_ipython().run_line_magic('run', 'rund2dir.py --dir ~/code/eegml/seizure-videos/myoclonic-atonic-vid02-dir')
get_ipython().run_line_magic('run', 'run2dir.py --dir ~/code/eegml/seizure-videos/myoclonic-atonic-vid02-dir')
get_ipython().run_line_magic('run', 'run_video2.py --help')
get_ipython().run_line_magic('run', 'run_video2.py --resolution 582x480 --showBG True --video ~/code/eegml/seizure-videos/myoclonic-atonic-vid02.mp4')
get_ipython().run_line_magic('run', 'run2.py --image ~/code/eegml/seizure-videos/myoclonic-atonic-vid02-dir/myoclonic-atonic-vid02.1.png')
get_ipython().run_line_magic('run', 'run_video2.py --showBG True --video ~/code/eegml/seizure-videos/myoclonic-atonic-vid02.mp4')
get_ipython().run_line_magic('run', 'run_video2.py --showBG True --video ~/code/eegml/seizure-videos/myoclonic-atonic-vid02.mp4')
get_ipython().run_line_magic('run', 'run_video2.py --showBG True --video ~/code/eegml/seizure-videos/myoclonic-atonic-vid02.mp4')
