import glob

from setuptools import find_packages, setup

package_name = 'loki_navigate_to_goal'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'wayPoints.txt']),
        ('share/' + package_name + '/launch', glob.glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Lab 4: drive through waypoints with odometry while avoiding obstacles',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'odom_viz_server = loki_navigate_to_goal.odom_viz_server:main',
            'vlio_node = loki_navigate_to_goal.vlio_node:main',
            'go_to_goal = loki_navigate_to_goal.go_to_goal:main',
        ],
    },
)
