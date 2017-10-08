setup(
    name='deep_learning_from_scratch',
    version='0.0.1',
    description='Sample package for python-Guide.org',
    long_description=readme,
    author='me',
    author_email='XXXX@gmail.com',
    install_requires=['numpy', 'matplotlib'],
    #dependancy_links=['git+ssh://git@github.com/our-company-internal/unko.git#egg=unko'],
    url='xxxx.com/',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    test_suite='tests'
)
