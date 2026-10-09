import glob

from setuptools import find_packages, setup

package_name = 'loki_chase_object'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Lab 3: chase a tracked object by combining camera bearing and LIDAR range',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'detect_object = loki_chase_object.detect_object:main',
            'get_object_range = loki_chase_object.get_object_range:main',
            'chase_object = loki_chase_object.chase_object:main',
        ],
    },
)
