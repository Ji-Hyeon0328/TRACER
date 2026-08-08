from setuptools import find_packages, setup


package_name = "tracer_pympc_bridge"


setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(
        exclude=["test"],
    ),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="JiHyun Yu",
    maintainer_email="prince31663@gmail.com",
    description=(
        "ROS2 UDP sidecar for the TRACER "
        "Quadruped-PyMPC runtime"
    ),
    license="MIT",
    entry_points={
        "console_scripts": [
            (
                "meta_gait_udp_bridge = "
                "tracer_pympc_bridge."
                "meta_gait_udp_bridge:main"
            ),
        ],
    },
)
