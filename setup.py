from setuptools import setup, find_namespace_packages

if __name__ == "__main__":
    setup(
        name="gan_project",
        description="Package used for a DL project about GANs",
        package_dir={"": "src"},
        author="Julien Mereau and Agathe Minaro",
        packages=find_namespace_packages('./src'),
        python_requires=">=3.9",
    )