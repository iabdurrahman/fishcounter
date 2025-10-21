## clone this REPO!

# 📦 HOW TO COMBINE MODELS

# 🖥️ Combining Models Locally (PC Setup)

## 🧪 1. System Requirements
in your PC :

- Check your NVIDIA version:

    ```bash
    nvidia-smi
    ```

- Install CUDA:  
  [Download CUDA 12.5](https://developer.nvidia.com/cuda-12-5-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)

- Install GPU-compatible PyTorch:  
  *(Personally using PyTorch 2.5.1)*

---

follow this github : 
```
git clone --depth 1 https://github.com/ArvinNathanielTjong/fishcounter-training.git
```


# 🍊 Orange Pi 5 Pro Setup

## Download official SD Card Image for Orange Pi 5 Pro

Get your latest image from:
http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/service-and-support/Orange-Pi-5-Pro.html

Currently we use ubuntu image (desktop XFCE, kernel 6.1.43), other image may work, but it not tested.

Flash your image to MicroSD card with minimum size 16 GB. You can flash your MicroSD card by tools like `dd` or balena etcher.

Optional: put your `authorized_keys` in your home ssh (`/root/.ssh`, `/home/orangepi/.ssh`) so you can ssh into orangepi immediately

## Update your system and install/download additional software package

to update your system, connect your orange pi to internet, and run this:
``` bash
sudo apt update
sudo apt upgrade --no-install-recommends
```

install python-tk (tk is not available in virtual environment, so use native from system)
```bash
sudo apt install python3-tk --no-install-recommends 
```

Install virtual environment & cmake. you can use `apt` to install those, but we will manually install locally in "$HOME/fakeroot/local" folder.

Download from here:
1. virtualenv - https://bootstrap.pypa.io/virtualenv.pyz
2. cmake - https://cmake.org/download/ (use the aarch64)

Put `virtualenv.pyz` in `fakeroot/local`. Extract cmake in `fakeroot/local`. You will have following result in `fakeroot/local` (in this example, we use cmake version 4.1.2):
```
orangepi@orangepi5pro:~/fakeroot/local$ ll
total 8204
drwxr-xr-x 3 orangepi orangepi    4096 Oct 21 21:44 ./
drwxr-xr-x 4 orangepi orangepi    4096 Oct 21 21:42 ../
drwxr-xr-x 6 orangepi orangepi    4096 Oct 21 21:43 cmake-4.1.2-linux-aarch64/
-rw-r--r-- 1 orangepi orangepi 8386700 Oct 11 05:33 virtualenv.pyz
```

(optional) make soft link `cmake` to `cmake-<version>-linux-aarch64` so we can refer to it easily.
```
orangepi@orangepi5pro:~/fakeroot/local$ ln --verbose --interactive --symbolic cmake-4.1.2-linux-aarch64 cmake
'cmake' -> 'cmake-4.1.2-linux-aarch64'
```

after link created, you will have `fakeroot/local` like this:
```
orangepi@orangepi5pro:~/fakeroot/local$ ll
total 8204
drwxr-xr-x 3 orangepi orangepi    4096 Oct 21 21:47 ./
drwxr-xr-x 4 orangepi orangepi    4096 Oct 21 21:42 ../
lrwxrwxrwx 1 orangepi orangepi      25 Oct 21 21:47 cmake -> cmake-4.1.2-linux-aarch64/
drwxr-xr-x 6 orangepi orangepi    4096 Oct 21 21:43 cmake-4.1.2-linux-aarch64/
-rw-r--r-- 1 orangepi orangepi 8386700 Oct 11 05:33 virtualenv.pyz
```

## Clone these repository to HOME directory

fish counter: as main application and needed for setup virtual environment
``` bash
git clone "https://github.com/iabdurrahman/fishcounter.git" --recurse-submodules
```

rknn-toolkit2:
```bash
git clone "https://github.com/airockchip/rknn-toolkit2.git" --recurse-submodules
```

## Replace `librknnrt.so` and `librknn_api.so` with the one from rknn-toolkit2 repository

backup current library:
```bash
sudo mv --verbose --interactive   "/usr/lib/librknnrt.so"     "/usr/lib/librknnrt.so~"
sudo mv --verbose --interactive   "/usr/lib/librknn_api.so"   "/usr/lib/librknn_api.so~"
```

copy library from repository:
```bash
sudo cp --verbose --interactive   "$HOME/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so"  "/usr/lib/librknnrt.so"
sudo cp --verbose --interactive   "$HOME/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so"  "/usr/lib/librknn_api.so"
```

## Copy C header file from rknn-toolkit2 repository to `/usr/include`

```bash
sudo cp --verbose --interactive   -t  "/usr/include" "$HOME/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/include"/*
```

## Create python virtual environment

create virtual environment in `"$HOME/venv3"`
```bash
python3 "$HOME/fakeroot/local/virtualenv.pyz" --download --python="python3" "$HOME/venv3"
```

activate virtual environment and add cmake to `$PATH`
```bash
. "$HOME/venv3/bin/activate"
export PATH="$PATH:$HOME/fakeroot/local/cmake/bin"
```

export cmake minimum as work around for `cmake_minimum_required`:
```bash
export CMAKE_POLICY_VERSION_MINIMUM="3.7"
```

install preliminary package from fishcounter repository:
```bash
pip install --verbose --require-virtualenv --requirement "$HOME/fishcounter/preliminary_requirements.txt"
```

install packages for rknn_toolkit2; check your python version and use the one match your python version:
```bash
orangepi@orangepi5pro:~$ python --version
Python 3.10.12
```

in this example, we will use `3.10` as version for installation
```bash
pip install --verbose --require-virtualenv --requirement "$HOME/rknn-toolkit2/rknn-toolkit2/packages/arm64/arm64_requirements_cp310.txt"
pip install --verbose --require-virtualenv "$HOME/rknn-toolkit2/rknn-toolkit2/packages/arm64/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
```

install requirements from fishcounter repository:
```bash
pip install --verbose --require-virtualenv --requirement "$HOME/fishcounter/requirements.txt"
```


## add udev rules and add group `gpio`

these rules are for connecting to arduino board (through uart) and access to gpio (for buzzer) as regular user (not root)

```bash
sudo cp --verbose --interactive -t "/etc/udev/rules.d" "$HOME/fishcounter/udev_rules"/*
```

add group `gpio` if not yet exist (you can check it in `/etc/group`)
```bash
sudo groupadd "gpio"
```

make current user (in this case is `orangepi`) included in group `gpio`

```bash
sudo usermod --append --groups "gpio" "orangepi"
```


## adapt platform dependent file to your platform

Use these template to produce file:
1. `$HOME/fishcounter/config.yaml.in` to create `$HOME/fishcounter/config.yaml`
2. `$HOME/fishcounter/launcher_app_wrapper.sh.in` to create `$HOME/fishcounter/launcher_app_wrapper.sh`
3. `$HOME/fishcounter/crontab.in` to create `$HOME/fishcounter/crontab`

set script `$HOME/fishcounter/launcher_app_wrapper.sh` to be executable:
```bash
chmod --verbose "ugoa+x" "$HOME/fishcounter/launcher_app_wrapper.sh"
```

## create crontab so application can startup automatically

```bash
sudo crontab -u "root" "$HOME/fishcounter/crontab"
```

# Test with X11 forwarding through SSH

after opening ssh session with X11 forwarding, run:

```bash
"$HOME/fishcounter/launcher_app_wrapper.sh" --forward
```


---

# 🔧 Tuning Detection & Tracking Sensitivity

Your fish counter's performance is controlled by six key parameters. Tweaking them can significantly change how accurately it detects and tracks fish. These settings are found in two different files.

## Part 1: Initial Detection (obj_thresh & nms_thresh)

These two parameters control the initial object detection performed by the AI model. They determine what the system considers a valid "fish" in a single frame, before tracking even begins.

➡️ Where to Edit: You can find these settings in the launcher_app.py file, inside the get_detector method.

``` bash
# In launcher_app.py

def get_detector(self):
    """Membuat instance detector jika belum ada."""
    if self.detector is None:
        try:
            # ...
            self.detector = ObjectDetector(model_path=model_path, 
            img_size=(640, 640), 
            obj_thresh=0.048,   # <-- EDIT HERE
            nms_thresh=0.048)   # <-- AND HERE
        # ...
```

### 1. obj_thresh (Object Threshold)
What it is: The minimum confidence score (from 0.0 to 1.0) the AI must have to consider a detection valid.

Analogy: Think of it as asking the AI, "How sure are you that this is a fish?" A value of 0.048 means the AI only needs to be 4.8% sure.

How to Tune:

Increase this value (e.g., to 0.3) if you are getting too many false positives (detecting things that aren't fish). This makes the AI more "picky."

Decrease this value if the AI is missing fish that are hard to see. This makes the AI less "picky."

---
### 2. nms_thresh (Non-Max Suppression Threshold)
What it is: The overlap threshold. It's used to clean up cases where the AI draws multiple bounding boxes on the same single fish.

Analogy: If the AI draws three boxes on one fish, NMS decides to keep only the "best" one and discard the others that overlap too much.

How to Tune:

Decrease this value (e.g., to 0.2) if single fish are being counted multiple times. This will more aggressively merge overlapping boxes.

Increase this value (e.g., to 0.6) if you have many fish very close together and the tracker is mistakenly merging them into one box.

---

## Part 2: Fine-Tuning the Tracking
These four parameters control the Sort tracker. They don't affect the initial detection; they affect how the system connects detections from one frame to the next to maintain a consistent ID for each fish.

➡️ Where to Edit: You can find these settings in the object_detector.py file, inside the __init__ method where mot_tracker is created.

``` bash
# In /utils/object_detector.py

class ObjectDetector:
    def __init__(self, ...):
        # ...
        self.mot_tracker = Sort(max_age=1,            # <-- EDIT HERE
        min_hits=1,           # <-- EDIT HERE
        diou_threshold=0.3,   # <-- EDIT HERE
        dij_threshold=0.9)    # <-- EDIT HERE
        # ...
```

### 3. max_age ⏳
What it is: The tracker's short-term memory. It's the max number of frames a track can exist without being matched to a detection before it's deleted.

Analogy: If a fish disappears behind an obstacle, this is how many frames the tracker will "remember" it and keep looking for it.

How to Tune: The current value of 1 is very low. Increase it (e.g., to 5 or 10) if fish that are momentarily lost are getting assigned new IDs when they reappear.

### 4. min_hits ✨
What it is: The confirmation rule. It's the number of consecutive frames a detection must appear before it's given a permanent track ID.

Analogy: This prevents a random, one-frame glitch from being counted as a fish. It needs to see the object a few times to be sure.

How to Tune: The current value of 1 means detections are trusted instantly. Increase it (e.g., to 2 or 3) if you are getting phantom tracks from noise or false detections.

### 5. dij_threshold 🎯
What it is: The strict matching rule for high-confidence detections. It requires a high similarity score (based on the distance between object centers) for a match.

Analogy: Matching a crystal-clear photo to a passport photo. It needs to be a near-perfect match.

How to Tune: A value of 0.9 is very strict. Decrease it slightly (e.g., to 0.85) if stable tracks are being lost because the fish moves too fast between frames.

### 6. diou_threshold 🖇️
What it is: The lenient matching rule for low-confidence detections. It uses a combined score of distance and box overlap.

Analogy: Matching a blurry security camera photo. You're more forgiving just to keep the track from being lost.

How to Tune: A value of 0.3 is fairly lenient. This is generally a good value, but you could lower it (e.g., to 0.2) to try and hold onto tracks even more tenaciously, at the risk of an incorrect match.
