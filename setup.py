from setuptools import find_packages, setup

package_name = 'task1_mission_rospossible'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mos3ad',
    maintainer_email='mos3ad@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'video_streamer = task1_mission_rospossible.video_streamer:main',
            'flow_estimation = task1_mission_rospossible.flow_estimation:main',
            'tracking = task1_mission_rospossible.tracking:main',
            'fusion_node = task1_mission_rospossible.fusion_node:main',
            'segmentation = task1_mission_rospossible.segmentation:main',
        ],
    },
)
