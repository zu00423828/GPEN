from setuptools import setup, find_packages



setup(
    name="gpen",
    version="0.0",
    # version=subprocess.check_output(
    #     ['git', 'describe', '--tags']).strip().decode('ascii'),
    url="https://github.com/yangxy/GPEN",
    packages=find_packages(include=['gpen']),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.7',

)