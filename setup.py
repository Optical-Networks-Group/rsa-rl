
from setuptools import setup,find_packages

def _requires_from_file(filename):
    return open(filename).read().splitlines()

setup(
    name='rsarl',
    version='1.0.0',
    license='NTT License',
    description='Deep Reinforcement Learning for RSA',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Masayuki Shimoda',
    author_email='moshimoshimoshida@gmail.com',
    url='https://github.com/Optical-Networks-Group/rsa-rl',
    install_requires=_requires_from_file('requirements.txt'),
    packages=find_packages(exclude=('tests', 'docs')),
    package_dir={'rsarl': 'rsarl'},
    package_data={'rsarl': ['visualizer/assets/*.css']},
    entry_points={
        "console_scripts": [
            "rsa-rl-visualizer = rsarl.visualizer.visualizer:run_cli"
        ]
    }

)
