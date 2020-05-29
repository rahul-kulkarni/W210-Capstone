import argparse
import json
import logging
import subprocess

from tqdm import tqdm
logger = logging.getLogger(__name__)


def main(experiment_info_path, test_set_uuid, host_worksheet_uuid):
    logger.info(f"Loading experiments from {experiment_info_path}")
    with open(experiment_info_path) as experiment_info_file:
        experiments = json.load(experiment_info_file)
    logger.info(f"Intended host worksheet: {host_worksheet_uuid}")

    # Print the current codalab status
    logger.info("CodaLab status:")
    print(subprocess.run(["cl", "status"],
                         stderr=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         universal_newlines=True).stdout)
    # Get the uuid of all the bundles on the host worksheet.
    logger.info("Getting bundles in host worksheet...")
    host_worksheet_bundle_uuids = subprocess.run(["cl", "search", f"host_worksheet={host_worksheet_uuid}",
                                                  ".limit=100000", "-u"],
                                                 stderr=subprocess.PIPE,
                                                 stdout=subprocess.PIPE,
                                                 universal_newlines=True).stdout.split("\n")
    host_worksheet_bundle_uuids = [uuid for uuid in host_worksheet_bundle_uuids
                                   if uuid.strip() != "" and uuid is not None]
    logger.info(f"Got information for {len(host_worksheet_bundle_uuids)} bundles in the host worksheet!")

    bundle_information = {}
    logger.info("Getting info for each bundle in host worksheet...")
    for host_worksheet_bundle_uuid in tqdm(host_worksheet_bundle_uuids):
        # Get the name of the bundle with that associated UUID
        host_worksheet_bundle_name = subprocess.run(
            ["cl", "info", host_worksheet_bundle_uuid, "-f", "name"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True).stdout.strip("\n")
        bundle_information[host_worksheet_bundle_uuid] = {
            "name": host_worksheet_bundle_name
        }
    logger.info(f"Got information for {len(bundle_information)} bundles in the host worksheet!")

    for experiment in tqdm(experiments):
        experiment_predictions_name = f"{experiment['name']}-predictions"
        # Check if a bundle with experiment_name-predictions exists.
        predictions_exist = False
        predictions_uuid = None
        for bundle_uuid, bundle_metadata in bundle_information.items():
            if experiment_predictions_name == bundle_metadata["name"]:
                # Get the state of the prediction bundle
                prediction_bundle_state = subprocess.run(
                    ["cl", "info", experiment_predictions_name, "-f", "state"],
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    universal_newlines=True).stdout.strip("\n")
                # If the prediction bundle is queued or running,
                # we can make the evaluation bundle.
                if prediction_bundle_state in (
                        "created", "starting", "preparing",
                        "running", "finalizing", "ready"):
                    predictions_exist = True
                    predictions_uuid = bundle_uuid
                    logger.info(f"Found non-failing prediction bundle "
                                f"for {experiment['name']}!")
                    break
                # If the prediction bundle failed, we don't want to
                # make the evaluation bundle from it.
                else:
                    logger.warn(f"Found failed prediction bundle for "
                                f"{experiment['name']}, will keep looking...")

        # If predictions don't exist, then skip.
        if not predictions_exist or predictions_uuid is None:
            logger.warn(f"Didn't find predictions for "
                        f"{experiment['name']}, skipping...")
            continue

        # Check if an experiment with the same name already
        # exists on the worksheet. If any experiment with the
        # same name is in the "created" or "ready" state,
        # then don't rerun.
        experiment_queued_or_succeeded = False
        for host_worksheet_bundle_uuid in bundle_information:
            host_worksheet_bundle_name = bundle_information[host_worksheet_bundle_uuid]["name"]
            if host_worksheet_bundle_name == f"{experiment['name']}-evaluation":
                # This bundle is a match, get the state
                host_worksheet_bundle_state = subprocess.run(
                    ["cl", "info", host_worksheet_bundle_uuid, "-f", "state"],
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    universal_newlines=True).stdout.strip("\n")
                if host_worksheet_bundle_state in (
                        "created", "starting", "preparing",
                        "running", "finalizing", "ready"):
                    # This experiment is already queued or running or
                    # successfully run on the host worksheet,
                    # move on to the next bundle.
                    logger.info(f"Bundle {host_worksheet_bundle_name}"
                                f"({host_worksheet_bundle_uuid}) has state "
                                f"{host_worksheet_bundle_state}, not "
                                f"rerunning evaluation of predictions with "
                                f"name {experiment['name']}.")
                    experiment_queued_or_succeeded = True
                    break

        if not experiment_queued_or_succeeded:
            # Run the evaluation script on the prediction bundle
            subprocess.run(["cl", "run",
                            "evaluate.py:0xbcd57bee090b421c982906709c8c27e1",
                            f"dev.json:{test_set_uuid}",
                            f'predictions.json:{predictions_uuid}',
                            'python evaluate.py dev.json predictions.json',
                            "-n", f'{experiment["name"]}-evaluation'])

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s " "- %(name)s - %(message)s", level=logging.INFO
    )
    parser = argparse.ArgumentParser(
        description=("Given a JSON file with details about each "
                     "experiment, including the UUID of the model predictions,"
                     "evaluate the model on the SQuAD dev set."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment-info-path",
        type=str,
        required=True,
        help=("Path to JSON with info about each experiment.")
    )
    parser.add_argument(
        "--test-set-uuid",
        type=str,
        required=True,
        help=("UUID of SQuAD1.1-formatted test set to evaluate against. "
              "SQuAD 1.1 dev is 0x8f29fe78ffe545128caccab74eb06c57")
    )
    parser.add_argument(
        "--host-worksheet-uuid",
        type=str,
        required=True,
        help=("Experiments host worksheet uuid.")
    )
    args = parser.parse_args()
    main(args.experiment_info_path,
         args.test_set_uuid,
         args.host_worksheet_uuid)
