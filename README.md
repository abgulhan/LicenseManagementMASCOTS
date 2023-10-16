# LicenseManagementMASCOTS

License and Accounting logs: [LINK](https://pennstateoffice365-my.sharepoint.com/:u:/g/personal/abg6029_psu_edu/EQ76ALmR44xCobVQHybB5GcBlx4N9qYBxmooLIX8s4ZeLg?e=rYLC4i)



## Instructions

### Processing License Logs
Run the following command:

python stat_pipeline.py --log_file [location of downloaded license data]/comsol.log_deidentified --name COMSOL --max_license_file ./max_licenses/comsol_max_licenses.txt --start [optional yyyy-mm-dd] --end [optional yyyy-mm-dd] --show_statistics False
**Outputs** 

*Processed license log file*: ./data/[name].csv

*1 day spaced forecast data*: processed/combined\_[name]/combined_1day.csv


### Forecasting Example Command

python forecast.py --data_file ./processed/combined_COMSOL/combined_1day.csv --start 2017-01-31 --end 2020-01-31 --target COMSOLGUI --freq D --remove_feat_thresh 0.9 --feature_select False --scaling False --use_dynamic_real True --pred_len 360 --context_len 360 --pred_count 1 --epochs 5 --model mqcnn --quantiles 0.05,0.5,0.95  --save_model [optional] --load_model [optional]

### Generate Simulation Data

python job_parse.py --processed_license_log ./data/COMSOL.csv --accounting_log_directory [location of downloaded accounting log] --output_file ./processed/simulator/comsol2015_jobs_2017-10 --start_date 2017-10-01 --end_date 2017-11-03

### Running Simulation

Set parameters in simulator_tests.py

*Command line arguments will be added later*

