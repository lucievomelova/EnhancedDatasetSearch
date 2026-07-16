# Cron job configuration

To run the data processing pipeline every 24 hours, `cron` can be used.
To run it every day at 03:00, open the cron tab using `crontab -e` 
and add the following cron job:

```sh
0 3 * * * cd <path-to-the-project>/EnhancedDatasetSearch && <path-to-the-project>/EnhancedDatasetSearch/venv/bin/python -m EnhancedNkodDatasetSearch.data_processing.main --config_path config.yaml >> <path-to-the-project>/EnhancedDatasetSearch/logs/run.log 2>&1
```

This cron job expects that a Python virtual environment is created in the `venv` folder in the project root folder.
