import codecs
import os
import sys
  
try:
    from setuptools import setup
except:
    from distutils.core import setup

  
def read(fname):
    return codecs.open(os.path.join(os.path.dirname(__file__), fname)).read()
  
  
  
NAME = "mlstacking"

PACKAGES = ["mlstacking",]
  
DESCRIPTION = "mlstacking is a Python module for sklearn-API friendly multi-layer stacking"
  
LONG_DESCRIPTION = read("README.md")
  
KEYWORDS = "stacking multi-layer ensemble-learning"
  
AUTHOR = "HaoWang"
  
AUTHOR_EMAIL = "hao.wang2@tendcloud.com"
  
URL = "https://github.com/TalkingData/MultiLayerStacking"
  
VERSION = "0.2.2"
  
LICENSE = "BSD 3-Clause License"
  
setup(
    name = NAME,
    version = VERSION,
    description = DESCRIPTION,
    long_description = LONG_DESCRIPTION,
    classifiers = [
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
    ],
    keywords = KEYWORDS,
    author = AUTHOR,
    author_email = AUTHOR_EMAIL,
    url = URL,
    license = LICENSE,
    packages = PACKAGES,
    include_package_data=True,
    zip_safe=True,
)