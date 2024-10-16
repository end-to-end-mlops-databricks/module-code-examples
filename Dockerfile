FROM databricksruntime/python:15.4-LTS 

ARG PROJECT_DIR=/project

RUN pip install uv==0.4.20

WORKDIR ${PROJECT_DIR}
COPY dist/house_price-0.0.1-py3-none-any.whl ${PROJECT_DIR}/

RUN uv pip install --python /databricks/python3 ${PROJECT_DIR}/house_price-0.0.1-py3-none-any.whl