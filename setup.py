from setuptools import setup

setup(
    name='LixiPy',
    url='https://github.com/OTTAA-Project/LixiPy',
    author='OTTAA Software Dev Team',
    author_email='juanma.lopez@ottaaproject.com',
    # Needed to actually package something
    packages=['lixipy'],
    # Needed for dependencies
    install_requires=['numpy', 'scipy', 'matplotlib', 'pandas', 'scikit-learn', 'tensorflow', 'tkinter'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='Packaged used for the Python Processing applied at the Lixi EEG Project',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)