import os
import argparse
import logging

import pandas as pd
from utils.utils import batch_extractor, get_pickle, save_pickle
from utils.path import Get_path

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Extract BERT embeddings and save to pickle")
parser.add_argument("--dataset", required=True, help="Dataset base name (without extension)")
parser.add_argument("--type", default="csv", choices=["csv", "pkl"], help="Input file type")
parser.add_argument("--model_ckpt", default="bert-base-uncased", help="Hugging Face model checkpoint")
parser.add_argument("--chunk_size", default=10000, type=int, help="Rows per chunk for embedding extraction")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size for the BERT model")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting embedding extraction run")
    logger.info("Parsed arguments: %s", vars(args))

    data_folder_path = Get_path.data_path
    input_ext = "csv" if args.type == "csv" else "pkl"
    data_file_path = os.path.join(data_folder_path, f"{args.dataset}.{input_ext}")

    logger.info("Loading data from %s", data_file_path)
    if args.type == "csv":
        df = pd.read_csv(data_file_path, index_col= 0)
    else:
        df = get_pickle(data_file_path)
    logger.info("Dataset loaded: %d rows, %d columns", *df.shape)

    # -------------------------------------------------------------------------
    # Embedding extraction
    # -------------------------------------------------------------------------
    logger.info(
        "Extracting embeddings with model=%s, chunk_size=%d, batch_size=%d",
        args.model_ckpt, args.chunk_size, args.batch_size
    )
    emb_df = batch_extractor(
        df=df,
        text_column="text",
        model_ckpt=args.model_ckpt,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size
    )
    logger.info("Embedding extraction completed")

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    output_file_path = os.path.join(data_folder_path, "final_data.pkl")
    save_pickle(emb_df, output_file_path)
    logger.info("Embeddings saved to %s", output_file_path)

    logger.info("Run completed successfully")
