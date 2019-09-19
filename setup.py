from distutils.core import setup


setup(
    name="pyoptimum",
    version="0.1",
    packages=['pyoptimum'],
    python_requires='>=3.4',

    # metadata
    author="Mauricio C. de Oliveira",
    author_email="mauricio@vicbee.net",

    description="Python library for optimize.vicbee.net optimize api",
    license="MIT",

    keywords=["Optimization", "Porfolio Optimization"],
    install_requires=[
          'requests',
    ],
    url="https://github.com/mcdeoliveira/pyoptimum",
    download_url="https://github.com/mcdeoliveira/pyoptimum/archive/v0.1.tar.gz",

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Other Audience',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.4',
    ],
)
