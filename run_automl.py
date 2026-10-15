from __future__ import annotations

import argparse
import os
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from autogluon.tabular import TabularPredictor
from dotenv import load_dotenv
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import wandb

# use run_experiments.sh to run multiple experiments


def transform(df, label, col_setting):
    """
    Transformation of input data : sort out the needed columns and remove any based on the col_setting
    """

    df = df[~df[label].isna()]
    df_with_col = df.copy()

    if col_setting == "withx":
        ignore_cols = ["heading"]
    elif col_setting == "nox":
        ignore_cols = ["heading", "x", "y"]
    cols = [x for x in df.columns.tolist() if x not in ignore_cols]

    working_cols = cols
    df = df[working_cols]

    return df, df_with_col


def standardize(train_data, test_data, label, type):
    """
    Standardize training and test using Sk-Learn

    train_data: the train data to be standardize
    train_data: the test data to be standardize
    label: the tree class or the height

    """

    ignore_cols = [label]

    # Remove the locational columns we don't want to standardize
    train_label = train_data[ignore_cols]
    test_label = test_data[ignore_cols]
    train_subset = train_data.drop(ignore_cols, axis=1)
    test_subset = test_data.drop(ignore_cols, axis=1)

    if type == "standard":
        scaler = StandardScaler()
    elif type == "minmax":
        scaler = MinMaxScaler()
    else:
        print(
            f"Value Error: {scaler} is incorrect argument. Select from 'minmax' or 'standard' as the standard_scaler"
        )
        return

    scaler.fit(train_subset)

    scaled_train_subset = scaler.transform(train_subset)
    scaled_test_subset = scaler.transform(test_subset)
    train_subset_df = pd.DataFrame(
        scaled_train_subset, index=train_subset.index, columns=train_subset.columns
    )
    test_subset_df = pd.DataFrame(
        scaled_test_subset, index=test_subset.index, columns=test_subset.columns
    )
    train_subset_df = pd.concat([train_subset_df, train_label], axis=1)
    test_subset_df = pd.concat([test_subset_df, test_label], axis=1)

    return train_subset_df, test_subset_df


def save_results(results, output_path):
    """
    Save results locally
    TODO now uing wandb - probs dont need to run this
    """
    res_string = str(results)
    with open(os.path.join(output_path, "model_summary.txt"), "w") as f:
        f.write(res_string)

def main(
    output_path,
    train_path,
    test_path,
    pol_union,
    train_subset_prop,
    time_limit,
    model_preset,
    label,
    dataset_name,
    col_setting,
    run_feature_importance,
    standarization_type,
    polarimetry,
    uuid_identifier,
):

    """
    Run autogluon automl training loop
    it logs the run to wandb: you need to store your wandb credentials (API KEY) in .env file (put in gitignore file)
    """
    # Log run settings to wandb
    wandb.config.update(
        {
            "pol_union": pol_union,
            "train_subset_prop": train_subset_prop,
            "time_limit": time_limit,
            "model_preset": model_preset,
            "label": label,
            "dataset_name": dataset_name,
            "col_setting": col_setting,
            "run_feature_importance": run_feature_importance,
            "standarization": standarization_type,
            "polarimetry": polarimetry,
        }
    )
    output_directory = (
        f"{output_path}/dname_{dataset_name}__target_{label}__timelim_{time_limit}__modelset_"
        f"{model_preset}__tsp_{train_subset_prop}__polarim_{polarimetry}__colset_{col_setting}__uuid_{uuid_identifier}"
    )

    # Check if model summary already exists
    # if os.path.exists(os.path.join(output_directory, 'model_summary.txt')):
    # print(f"Model summary already exists for {output_directory}. Skipping this run.")
    # return

    df = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    if pol_union == "no":
        print("Running with separate polarimetry")
        polarimetry = int(polarimetry)
        if polarimetry not in [0, 1, 2]:
            raise ValueError("Polarimetry setting must be 0,1,2")
        train_data = df[df["polarimetry"] == polarimetry].copy()
        test_data = test_data[test_data["polarimetry"] == polarimetry].copy()
    else:
        print("Running with polarimetry unified")
        train_data = df.copy()

    if train_subset_prop > 1:
        raise ValueError("Train subset must be less than 1 ")

    if train_subset_prop != 1:
        train_subset, _ = train_test_split(
            train_data, test_size=1 - train_subset_prop, random_state=42
        )
    else:
        train_subset = train_data

    train_subset, train_subset_with_coord = transform(train_subset, label, col_setting)
    size_train = len(train_subset)
    test_data, test_subset_with_coord = transform(test_data, label, col_setting)

    # x and y values before saving
    train_coordinates = train_subset_with_coord[["x", "y"]].rename(
        columns={"x": "original_x", "y": "original_y"}
    )
    test_coordinates = test_subset_with_coord[["x", "y"]].rename(
        columns={"x": "original_x", "y": "original_y"}
    )

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    train_subset, test_data = standardize(
        train_subset, test_data, label, args.standarization
    )
    print(f"train_subset min: {train_subset.min()}, max{train_subset.max()}")
    print(f"test_data min: {test_data.min()}, max{test_data.max()}")

    predictor = TabularPredictor(label, path=output_directory).fit(
<<<<<<< HEAD
        train_subset,
        time_limit=time_limit,
        presets=model_preset, num_gpus = 1,
        ag_args_fit={'num_gpus': 1}
=======
        train_subset, time_limit=time_limit, presets=model_preset, num_gpus=1
>>>>>>> main
    )

    y_pred_test = predictor.predict(test_data.drop(columns=[label]))
    results = predictor.evaluate_predictions(
        y_true=test_data[label], y_pred=y_pred_test, auxiliary_metrics=True
    )
    size_test = len(test_data)

    # save results
    print(results)
    wandb.log(results)
    sizett = {"len_train": size_train, "len_test": size_test}
    results.update(sizett)
    wandb.log({"train_size": size_train, "test_size": size_test})
    save_results(results, output_directory)

    # Save the predictions to CSV including the locational data
    predictions_test_df = test_data.copy()  # preserve x and y
    predictions_test_df["predicted"] = y_pred_test
    predictions_test_df = pd.concat([predictions_test_df, test_coordinates], axis=1)

    predictions_test_df.to_csv(
        os.path.join(output_directory, "predictions_test.csv"), index=False
    )

    y_pred_train = predictor.predict(train_subset.drop(columns=[label]))
    predictions_train_df = train_subset.copy()
    predictions_train_df["predicted"] = y_pred_train
    predictions_train_df = pd.concat([predictions_train_df, train_coordinates], axis=1)
    predictions_train_df.to_csv(
        os.path.join(output_directory, "predictions_train.csv"), index=False
    )

    # run model leaderboard and save
    res = predictor.leaderboard(test_data)
    res.to_csv(os.path.join(output_directory, "leaderboard_results.csv"), index=False)
    # Log the leaderboard to wandb
    wandb.log({"leaderboard": wandb.Table(dataframe=res)})

    # Log confusion matrix if the task is classification
    if label == "class":
        cm = confusion_matrix(test_data[label], y_pred_test)

        # Normalize the confusion matrix and plot percentages
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_percentage = cm_normalized * 100  # Convert to percentage

        # Plot confusion matrices side by side
        fig, axes = plt.subplots(1, 2, figsize=(20, 7))

        # Confusion matrix with absolute values
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=predictor.class_labels,
            yticklabels=predictor.class_labels,
            ax=axes[0],
        )
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")
        axes[0].set_title("Confusion Matrix (Absolute Values)")

        # Confusion matrix with percentages
        sns.heatmap(
            cm_percentage,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=predictor.class_labels,
            yticklabels=predictor.class_labels,
            ax=axes[1],
        )
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("True")
        axes[1].set_title("Confusion Matrix (Percentages)")

        # Save the figure and log to wandb
        confusion_matrix_fig_path = os.path.join(
            output_directory, "confusion_matrices.png"
        )
        fig.savefig(confusion_matrix_fig_path)
        wandb.log({"confusion_matrices": wandb.Image(confusion_matrix_fig_path)})
        plt.close(fig)

        # Calculate and log micro precision and recall
        precision, recall, _, _ = precision_recall_fscore_support(
            test_data[label], y_pred_test, average="micro"
        )
        wandb.log({"micro_precision": precision, "micro_recall": recall})

        # Print classification report
        unique_classes_true = np.unique(test_data[label])
        target_names = [str(cls) for cls in unique_classes_true]
        class_report = classification_report(
            test_data[label], y_pred_test, target_names=target_names
        )
        print("Classification Report:\n", class_report)
        wandb.log({"classification_report": class_report})

    if run_feature_importance is True:
        pred = predictor.feature_importance(test_data)
        pred.to_csv(
            os.path.join(output_directory, "feature_importance.csv"), index=False
        )


if __name__ == "__main__":
    # Load the .env file
    load_dotenv()
    # Retrieve the wandb API key from the environment variable (in .env file)
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)

    parser = argparse.ArgumentParser(description="Run AutoGLuon AutoML Script")
<<<<<<< HEAD
    
    parser.add_argument('--time_limit', default=60, type=float, help='Time limit for the AutoML training')
    parser.add_argument('--model_preset', default="best_quality",type=str, help='Model preset to use e.g. medium_quality for prototpying, best_quality for final run ')
    parser.add_argument('--train_subset_prop',default=1.0, type=float, help='Proportion of the training subset')
    parser.add_argument('--output_path', default="/home/laura93schade/local/output", type=str, help='Path to save the output')
    parser.add_argument('--pol_union', default="no", type=str,  help='Polarimetry union, yes or no')
    parser.add_argument('--input_folder', default="/home/laura93schade/local/mldata", type=str, help='Input data folder')
    parser.add_argument('--geo_name', default="square", type=str, help='Geographical split name, split or swath')
    parser.add_argument('--label' , default="lidar", type=str, help='label of target column for ML: class or lidar')
    parser.add_argument('--band_opt', default='P',  help='band selection for the data-generation (P, L, P+L)')
    parser.add_argument('--standarization', default='scaler',  help='type of standarization scaler or minmaxscaler')
=======

    parser.add_argument(
        "--time_limit",
        default=60,
        type=float,
        help="Time limit for the AutoML training",
    )
    parser.add_argument(
        "--model_preset",
        default="best_quality",
        type=str,
        help="Model preset to use e.g. medium_quality for prototpying, best_quality for final run ",
    )
    parser.add_argument(
        "--train_subset_prop",
        default=1.0,
        type=float,
        help="Proportion of the training subset",
    )
    parser.add_argument(
        "--output_path",
        default="/home/laura93schade/local/models",
        type=str,
        help="Path to save the output",
    )
    parser.add_argument(
        "--pol_union", default="no", type=str, help="Polarimetry union, yes or no"
    )
    parser.add_argument(
        "--input_folder",
        default="/home/laura93schade/local/mldata",
        type=str,
        help="Input data folder",
    )
    parser.add_argument(
        "--geo_name",
        default="square",
        type=str,
        help="Geographical split name, split or swath",
    )
    parser.add_argument(
        "--label",
        default="class",
        type=str,
        help="label of target column for ML: class or lidar",
    )
    parser.add_argument(
        "--band_opt",
        default="P",
        help="band selection for the data-generation (P, L_Mono, L_Bi, P+L)",
    )
    parser.add_argument(
        "--standarization",
        default="scaler",
        help="type of standarization scaler or minmaxscaler",
    )
>>>>>>> main
    args = parser.parse_args()

    run_feature_importance = False

    dataset_name = "geo_split_" + args.geo_name + "__polunion:_" + args.pol_union
    run_name = f"{dataset_name}_tsp{args.train_subset_prop}_target{args.label}_time{args.time_limit}_preset{args.model_preset}_{args.standarization}"

    if args.geo_name == "swath":
        thick = 0.2
    elif args.geo_name == "square":
        thick = 0.05
    
    if args.pol_union == "yes":
        train_path = f"{args.input_folder}/{args.geo_name}_{args.label}_mldata_union_pol__train_data__thick_{thick}__testsize_0.2.csv"
        test_path = f"{args.input_folder}/{args.geo_name}_{args.label}_mldata_union_pol__test_data__thick_{thick}__testsize_0.2.csv"

        polarimetry = None

        for col_setting in ["withx", "nox"]:
            uuid_identifier = str(uuid.uuid4())[0:9]
            run_name = run_name
            with wandb.init(
                entity="fdl-biomass",
                project=f"automl_{args.band_opt}_band_{args.label}",
                sync_tensorboard=True,
                reinit=True,
                name=run_name,
            ):
                # Initialize a new wandb run for each iterationeset
                main(
                    args.output_path,
                    train_path,
                    test_path,
                    args.pol_union,
                    args.train_subset_prop,
                    args.time_limit,
                    args.model_preset,
                    args.label,
                    dataset_name,
                    col_setting,
                    run_feature_importance,
                    args.standarization,
                    polarimetry,
                    uuid_identifier,
                )
    else:
        train_path = f"{args.input_folder}/{args.geo_name}_{args.label}_mldata_separate_pol__train_data__thick_{thick}__testsize_0.2.csv"
        test_path = f"{args.input_folder}/{args.geo_name}_{args.label}_mldata_separate_pol__test_data__thick_{thick}__testsize_0.2.csv"
        for polarimetry in [0, 1, 2]:
            for col_setting in ["withx", "nox"]:
                # Initialize a new wandb run for each iteration
                uuid_identifier = str(uuid.uuid4())[0:9]
                run_name = run_name
                with wandb.init(
                    entity="fdl-biomass",
                    project=f"automl_{args.band_opt}_band_{args.label}",
                    sync_tensorboard=True,
                    reinit=True,
                    name=run_name,
                ):
                    main(
                        args.output_path,
                        train_path,
                        test_path,
                        args.pol_union,
                        args.train_subset_prop,
                        args.time_limit,
                        args.model_preset,
                        args.label,
                        dataset_name,
                        col_setting,
                        run_feature_importance,
                        args.standarization,
                        polarimetry,
                        uuid_identifier,
                    )
