#reposible to build the app as package and install is as pipPy
#When -e . is added to requirements.txt, it tells pip to look for the setup.py file in the current directory (indicated by the .) and install the package described by that file in "editable mode."
#pip install -e . (can used to run setup.py and build editable package)

from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name = 'mlproject',
    version='0.0.1',
    author='Qandos',
    author_email='nooramerq0@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)