# Run this py to test if gpu is available

import tensorflow as tf
from tensorflow.python.client import device_lib

if __name__ == "__main__":
    print(f'[INFO] GPU AVAILABLE :{tf.test.is_gpu_available()}')
    print(f'[INFO] AVAILABLE DEVICES :{device_lib.list_local_devices()}')