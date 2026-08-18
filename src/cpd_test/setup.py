from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'cpd_test'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['numpy', 'pycpd', 'setuptools'],
    zip_safe=True,
    maintainer='gj64',
    maintainer_email='mathewmurray2005@outlook.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'cpd_node = cpd_test.cpd_node:main'
        ],
    },
)
