# Evaluation
This directory contains the evaluation script and a configuration file used for objective evaluation 
of the search platform. The evaluation script can be run by executing the following command from the project root folder:

```sh
python -m evaluation.grid_search  --config_path evaluation/config_grid_search.yaml
```

To see the results, start the `mlflow` server:
```sh
mlflow server --port 5001
```
and open the following URL in a web browser: `http://localhost:5001`. 
