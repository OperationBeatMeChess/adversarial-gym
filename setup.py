from setuptools import setup, find_packages

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='adversarial_gym',
      version='0.0.1',
      description='OpenAI Gym environments for adversarial games for the operation beat ourselves organisation.',
      url='https://github.com/OperationBeatMeChess/adversarial-gym',
      author='Dawson Horvath',
      author_email='horvath.dawson@gmail.com',
      license='MIT License',
      install_requires=['gym>=0.26.1', 'python-chess', 'numpy', 'cairosvg', 'pillow', 'pygame'],
      long_description=long_description,
      long_description_content_type="text/markdown",
)
