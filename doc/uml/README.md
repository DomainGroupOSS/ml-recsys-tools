1. ### make sure dependencies are installed pylint is installed by running
 
```bash
pip install pylint
sudo apt-get install graphviz
```

2. ### generage umls by:
 
`pyreverse -o png -p some_name_suffix package/subpackage/module` 

e.g.:  `pyreverse -o png -p recommeders ml_recsys_tools/recommenders`