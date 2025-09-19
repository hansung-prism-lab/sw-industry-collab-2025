import argparse
import logging
import os

from utils.utils import *
from utils.model import *

from sklearn.metrics import accuracy_score

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Multi-output text classifier")
parser.add_argument("--dataset", required=True, help="Base name of the pickled dataframe (without extension)")
parser.add_argument("--dropout", default=0.3, type=float, help="Dropout rate for hidden layers")
parser.add_argument("--learning_rate", default=0.0015, type=float, help="Initial learning rate")
parser.add_argument("--validation_split", default=0.125, type=float, help="Validation split ratio")
parser.add_argument("--batch_size", default=32, type=int, help="Mini-batch size")
parser.add_argument("--epochs", default=100, type=int, help="Training epochs")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting training run")
    logger.info("Parsed arguments: %s", vars(args))

    file_path = os.path.join(Get_path.data_path, f"{args.dataset}.pkl")
    encoder_path = Get_path.encoder_path
    logger.info("Loading data from %s", file_path)

    df = get_pickle(file_path)
    logger.info("Dataset loaded: %d rows, %d columns", *df.shape)

    drop_cols = ["text", "EMBEDDING"]
    y_col_list = [col for col in df.columns if col not in drop_cols]
    logger.debug("Target columns: %s", y_col_list)

    # -------------------------------------------------------------------------
    # Data partitioning
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = data_partition(
        fpath=file_path,
        epath=encoder_path,
        y_col=y_col_list
    )
    logger.info("Data partitioned: X_train=%s, X_test=%s", X_train.shape, X_test.shape)

    # -------------------------------------------------------------------------
    # Model creation
    # -------------------------------------------------------------------------
    output_info = make_output_info(y_train)
    y_train_dict = split_y_targets(y_train, output_info)
    logger.info("Output information: %s", output_info)

    model = create_model(
        output_info=output_info,
        dropout_rate=args.dropout,
        learning_rate=args.learning_rate
    )
    model.summary(print_fn=lambda x: logger.info(x))

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    logger.info("Training started (epochs=%d, batch_size=%d)", args.epochs, args.batch_size)
    trained_model = train_model(
        model=model,
        X_train=X_train,
        y_train_dict=y_train_dict,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.validation_split
    )
    logger.info("Training completed")

    # -------------------------------------------------------------------------
    # Inference & evaluation
    # -------------------------------------------------------------------------
    logger.info("Running inference on %d test samples", len(X_test))
    pred = model.predict(X_test, verbose=0)
    pred_dict = process_predictions(pred, output_info)

    logger.info("Per-target accuracy:")
    for key in pred_dict.keys():
        true_vals = y_test[key].values
        pred_vals = pred_dict[key]
        acc = accuracy_score(true_vals, pred_vals)
        logger.info("%s: %.4f", key, acc)

    logger.info("Training run completed successfully")
