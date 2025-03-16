from setuptools import find_packages , setup
from typing import List

def get_requirements(filepath) -> List[str]:
    
    requirements = []
    try:
        with open(filepath , 'r') as file_obj:
            lines = file_obj.readlines()

            for requirement in lines:
                requirement = requirement.strip()
                if requirement and requirement != '-e .':
                    requirements.append(requirement)
    except FileNotFoundError:
        print("File not found")

    return requirements       

setup(
    name="Network Security Project",
    version = '0.0.1',
    author = 'Uday Khunt',
    author_email = 'udaykhunt02@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)