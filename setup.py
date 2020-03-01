import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


setup(
    name="aprendizaje_automatico_aplicado",
    version="0.0.1",
    url="https://github.com/borntyping/cookiecutter-pypackage-minimal",

    author="Daniel Rodriguez Perez",
    author_email="daniel.rodriguez.perez@gmail.com",

    description="Complemento para las clases de Aprendizaje Automatico Aplicado",
    long_description=read("README.rst"),

    packages=find_packages(exclude=('tests')),

    install_requires=['matplotlib', 'numpy', 'pandas', 'scipy', 'sklearn'],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
