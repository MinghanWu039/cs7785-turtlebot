from setuptools import find_packages, setup

package_name = 'loki_nav2_waypoints'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'waypoints.txt']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Travis',
    maintainer_email='traviswu0524@gmail.com',
    description='Lab 5: send waypoints from waypoints.txt to Nav2 one at a time',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'waypoint_publisher = loki_nav2_waypoints.waypoint_publisher:main',
        ],
    },
)
