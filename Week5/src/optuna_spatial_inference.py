import optuna
from optuna_function import optuna_function

def objective(trial):
    num_clips=trial.suggest_categorical(
        "num_clips", [1,2,3,4,5]
    )
    crop_1=trial.suggest_categorical(
        "crop_1", [True,False]
    )
    crop_2=trial.suggest_categorical(
        "crop_2", [True,False]
    )
    crop_3=trial.suggest_categorical(
        "crop_3", [True,False]
    )
    crop_4=trial.suggest_categorical(
        "crop_4", [True,False]
    )
    crop_5=trial.suggest_categorical(
        "crop_5", [True,False]
    )
        
    # cridar a multiview_spatial_inference.py al main , passant num_clips com a input
    # return accuracy
    acc=0
    try:
        acc=optuna_function(num_clips,crop_1,crop_2,crop_3,crop_4,crop_5)
    except:
        print('An error ocurred')
    return acc

search_space = {
    "num_clips": [1,2,3,4,5],
    "crop_1": [True,False],
    "crop_2": [True,False],
    "crop_3": [True,False],
    "crop_4": [True,False],
    "crop_5": [True,False]
    
}

study = optuna.create_study(
    sampler=optuna.samplers.RandomSampler(seed=42),
    direction="maximize",  # redundand, since grid search
    storage="sqlite:///multiview_spatial_inference.db",
    study_name="study_final",
)

study.optimize(objective,n_trials=100)