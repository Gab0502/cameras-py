from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import subprocess

def get_ffmpeg_flags():
    """Obtém flags de compilação do FFmpeg"""
    try:
        cflags = subprocess.check_output(
            ['pkg-config', '--cflags', 'libavformat', 'libavcodec', 'libavutil']
        ).decode().strip().split()
        
        ldflags = subprocess.check_output(
            ['pkg-config', '--libs', 'libavformat', 'libavcodec', 'libavutil']
        ).decode().strip().split()
        
        return cflags, ldflags
    except:
        return [], ['-lavformat', '-lavcodec', '-lavutil']

cflags, ldflags = get_ffmpeg_flags()

ext_modules = [
    Pybind11Extension(
        "fast_video_merger",
        ["ffmpeg_cpp_wrapper.cpp"],
        extra_compile_args=['-std=c++17', '-O3', '-fopenmp'] + cflags,
        extra_link_args=['-fopenmp'] + ldflags,
    ),
]

setup(
    name="fast_video_merger",
    version="1.0.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)