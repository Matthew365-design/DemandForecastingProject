import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.test.is_gpu_available()}")
print(f"CUDA built: {tf.test.is_built_with_cuda()}")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"GPUs found: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
else:
    print("No GPUs found")