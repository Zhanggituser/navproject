from setuptools import find_packages, setup

package_name = 'rl_nav_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
         ('share/' + package_name + '/launch', ['launch/rl_nav2_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yuzhang',
    maintainer_email='qq1336146270@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'rl_controller_node = rl_nav_controller.rl_controller_node:main'
            'train_rl_nav = rl_nav_controller.train_rl_nav:train_rl',

        ],
    },
)
