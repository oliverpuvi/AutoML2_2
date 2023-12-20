# A decision support system for explainability techniques

The objective of this project is to create an automated decision support system designed to select interpretable methods tailored to specific dataset attributes and quantifiable evaluation metrics. With the growing intricacy of machine learning models, there's a pressing need for transparent, understandable explanations to facilitate informed decision-making. This system is intended to improve the ease of access to interpretability by suggesting the most appropriate method for each dataset, thereby fostering the development of robust and comprehensible machine learning models.

## Repository structure

The testing and metric generation for interpretability techniques can be found in the notebooks lime_works.ipynb, anchor_works.ipynb, and ciu_works.ipynb. The full pipeline and recommendation example is in the file full_pipeline_recommendation.ipynb.

The datasets records_lime.csv, records_anchor.csv, and records_ciu.csv include all metrics from the training datasets. Metadata_merged.csv includes all metadata from the training data featrures. 

## References
- Used OpenML datasets are cloned from this repositorium for easier access: https://github.com/Krish2208/InterpretabilityBenchmark
- Some of our metric calculation is based on metric calculation described in this repositorium: https://github.com/Krish2208/explanability/tree/main

