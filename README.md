# models-anne

Repo for Anne's models and code. Same setup as production.

## Model Setup

To implement a new model `alphafoo`, write a new Python script `alphafoo.py` as follows
* Write a function `get_model()` that returns a trained model object with a `predict()` method like a scikit-learn model.
* Use the `driver` module's `main()` to invoke `get_model()`.
    
```
if __name__ == "__main__": 
    sys.exit(main(get_model))
```

Then
* In `docker-compose.yml`, copy the dev/prod container setup of an existing model including secrets and volumes.
* Add `secrets/alphafoo-model_id.txt` with the model_id from https://numer.ai/models.
* Add `secrets/alphafoo-model_name.txt` with `alphafoo` as its contents.

**Do not commit the secrets to the repository.**

Test the model with

```
docker-compose build
docker-compose run alphafoo-dev scripts/run-prediction.sh
```

This script will invoke the `predict()` function of the model from `get_model()` with the training data, validation data, and live data.
The first two will show the rank correlations as computed by Numerai.
The last one will be saved for submission.

When you are satisfied with the model performance, you can submit predictions with

```
docker-compose build
docker-compose run alphafoo-prod scripts/run-submission.sh
```
