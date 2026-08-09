from setuptools import find_packages, setup


package_name = "tracer_highlevel"


setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
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
    maintainer="TRACER",
    maintainer_email="noreply@example.com",
    description=(
        "TRACER high-level ROS2 command boundary."
    ),
    license="MIT",
    entry_points={
        "console_scripts": [
            (
                "fixed_meta_gait_node = "
                "tracer_highlevel.fixed_meta_gait_node:"
                "main"
            ),
            (
                "scheduled_meta_gait_node = "
                "tracer_highlevel.scheduled_meta_gait_node:"
                "main"
            ),
            (
                "structural_meta_gait_node = "
                "tracer_highlevel.structural_meta_gait_node:"
                "main"
            ),
        ],
    },
)
