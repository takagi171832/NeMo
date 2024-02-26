# setup.py
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

# Cythonモジュールの設定
extensions = [
    Extension(
        name="beamsearch_cython",  # コンパイルされたモジュールの名前
        sources=["beamsearch_cython.pyx"],  # Cythonソースファイル
        include_dirs=[numpy.get_include()]  # NumPyのヘッダーファイルのパス
    )
]

setup(
    name="My BeamSearch Module",
    ext_modules=cythonize(extensions)
)
