from distutils.core import setup
from setuptools import find_packages

setup(
    name='pnlp',
    version='0.0.1',
    description='',
    url='',
    author='Practical NLP',
    author_email='',
    license='unlicensed',
    packages=find_packages(),
    package_dir={'pnlp':'pnlp'},
    package_data={'pnlp': ['pnlp/config.json', 'README.md']},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'pnlp = pnlp.__main__:main',
        ],
    },
    zip_safe=False,
    install_requires=[
        'pymongo==3.10.1',
        'cloudpickle==1.2.2',
        'umap-learn==0.4.1',
        'scikit-learn==0.22.2.post1',
        'sacred==0.8.1',
        'pandas==0.25.1',
        'seaborn==0.9.0',
        'fasttext==0.9.1',
        'beautifulsoup4==4.8.0',
        'nltk==3.4.5',
        'numpy==1.16.5',
        'hdbscan==0.8.26',
        'lxml==4.6.5',
        'dnspython==1.16.0',
        'pyspellchecker==0.5.4',
        'annoy==1.16.3',
        'gensim==3.8.1',
        'keras==2.3.1',
        'tensorflow==2.0.1',
        'transformers==3.1.0',
        'scipy==1.4.1',
        'pyenchant==3.1.1'
    ]
)
