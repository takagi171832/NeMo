# setup.py
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

# NumPyのヘッダーファイルへのパスを取得
numpy_include_dir = numpy.get_include()

# 拡張モジュールの設定
extensions = [
    Extension("cy_unigram_lm", ["cy_unigram_lm.pyx"], include_dirs=[numpy_include_dir])
]

setup(
    name="Fast One-gram Language Model",
    ext_modules=cythonize(extensions)
)
