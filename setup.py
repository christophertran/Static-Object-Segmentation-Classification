from setuptools import setup

package_name = 'Static-Object-Segmentation-Classification'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='christophertran',
    maintainer_email='ctran457@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'static_object_segmentation_node = Static-Object-Segmentation-Classification.static_object_segmentation_node:main'
        ],
    },
)
