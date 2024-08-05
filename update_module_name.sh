#!/bin/bash
# Mac users: You [may first need to install gnu-sed](https://github.com/MigleSur/GenAPI/issues/8)

module_name="pyiron_IntendedModuleName"
rst_delimit="========================="   # This should be as many '=' as the name length.

for file in .binder/postBuild \
            .github/ISSUE_TEMPLATE/*.md \
            docs/conf.py \
            docs/index.rst \
            notebooks/example.ipynb \
            elaston/_version.py \
            tests/unit/test_tests.py \
            .coveragerc \
            .gitattributes \
            MANIFEST.in \
            pyproject.toml
do
  sed -i "s/elaston/${module_name}/g" ${file}
  sed -i "s/======================/${rst_delimit}/g" ${file}
done


mv elaston ${module_name}

rm update_module_name.sh
