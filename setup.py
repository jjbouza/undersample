setuptools.setup(
    name="undersample", # Replace with your own username
    version="0.1.0",
    author="Jose Bouza",
    author_email="josejbouza@gmail.com",
    description="Tools for synthetic undersampling of dMRI data.",
    url="https://github.com/jjbouza/undersample-dmri",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'scipy',
        'dipy',
        'matplotlib'
    ],
    python_requires='>=3.6',
)
