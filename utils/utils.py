import pickle, os, torch
import pandas as pd, numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from transformers import AutoTokenizer, AutoModel
from tensorflow.keras.utils import to_categorical



def get_pickle(fpath: str):
    with open(fpath, "rb") as f:
        pkl = pickle.load(f)
    return pkl

def save_pickle(file: object, output_path: str): 
    with open(output_path, 'wb') as f:
        pickle.dump(file, f)

# -----------------------------------------------------------------------------
# bert_main.py
# -----------------------------------------------------------------------------
def embedding_extractor(data: pd.DataFrame, text_column: str, model_ckpt: str, batch_size: int) -> list[list[float]]:
# function of extracting [CLS] Token embedding from BERT-based model

    """
    model_ckpt: verion of BERT-based model
    batch_size: recommend that the value of this variable be 2 or 4
    """

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = AutoModel.from_pretrained(model_ckpt).to(device)

    embeddings = []
    text_list = data[text_column].tolist()

    for i in tqdm(range(0, len(text_list), batch_size)):
        batch_texts = text_list[i:i+batch_size]

        inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding="max_length", max_length = 512) # default of max_length is 512
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings.append(embedding.last_hidden_state[:, 0, :])  # append CLS token embedding data

    # Stack embeddings into a tensor
    stacked_embeddings = torch.cat(embeddings, dim=0)

    stacked_embeddings = stacked_embeddings.cpu().numpy()

    result = stacked_embeddings.tolist()

    return result

def batch_extractor(df: pd.DataFrame, text_column: str, chunk_size: int, model_ckpt: str, batch_size: int) -> pd.DataFrame:
    df_list = []
    len_df = len(df)
    for start_idx in range(0, len(df), chunk_size):
        end_idx = min(start_idx + chunk_size, len_df)
        temp_df = df.iloc[start_idx:end_idx]
        temp_df["EMBEDDING"] = embedding_extractor(temp_df, text_column, model_ckpt, batch_size = batch_size)
        df_list.append(temp_df)
        torch.cuda.empty_cache()

    result_df = pd.DataFrame(columns=df.columns)
    for temp_df in df_list:
        result_df = pd.concat([result_df, temp_df], axis = 0)

    return result_df

# -----------------------------------------------------------------------------
# main.py
# -----------------------------------------------------------------------------
def data_partition(fpath: str, epath: str, y_col: list[str])-> dict[str, np.ndarray | pd.DataFrame]:
    '''
    Label encoding, train/test split, one-hot encoding, and return 
    '''
    df = pd.read_pickle(fpath)
    df = df.reset_index()
    X = np.array(df["EMBEDDING"].tolist())

    # label encoding
    for col in y_col:
        df[col] = label_encoding(df, col, epath) 

    y_data = df[y_col].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=0.2) # split

    # One hot
    y_train = one_hot_encoder(y_train, y_col)

    return X_train, X_test, y_train, y_test


def label_encoding(df: pd.DataFrame, col: str, epath: str) -> pd.Series:
    '''
    Label encoding, save encoders, and return encoded result
    '''
    le = LabelEncoder()
    encoded_val =  le.fit_transform(df[col].values)

    encoder_file_path = os.path.join(epath, f"{col}.pkl")
    save_pickle(le, encoder_file_path)

    return encoded_val


def one_hot_encoder(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    ''' 
    One-hot encode the given columns and return the encoded DataFrame
    '''
    onehot_df_list = []

    for col in cols:
        arr = to_categorical(df[col])
        col_names = [f"{col}_{i}" for i in range(arr.shape[1])]
        onehot_df = pd.DataFrame(arr, columns=col_names)
        onehot_df_list.append(onehot_df)

    return pd.concat(onehot_df_list, axis=1)


def make_output_info(y: pd.DataFrame)-> list[dict[str, str]]:
    '''
    Return list of dicts with column names and node counts for model outputs.
    -> Used in create_model() to define output layers

    Example
    [{'name': 'GENDER', 'nodes': 2}, {'name': 'AGE', 'nodes': 6}]
    '''
    output_info = []

    added_names = set()  

    for col in y.columns:
        # AGE_1 -> AGE
        base_name = col.split('_')[0]

        if base_name not in added_names:
            # Count numbers of columns starting with base_name --> node counts
            class_count = sum(c.startswith(base_name.upper() + "_") for c in y.columns)
            output_info.append({
                'name': base_name,
                'nodes': class_count
            })
            added_names.add(base_name)

    return output_info


def split_y_targets(y_train: pd.DataFrame, output_info: list[dict[str, str]])-> dict[str, pd.DataFrame]:
    '''
    Split One hot encoded Dataframe by category {Category name : Dataframe}

    Example
    {'AGE':   AGE_0  AGE_1  AGE_2  AGE_3  AGE_4  AGE_5
     0        0.0    1.0    0.0    0.0    0.0    0.0
     1        0.0    0.0    1.0    0.0    0.0    0.0}
    '''
    y_dict = {}

    for info in output_info:
        name = info['name']
        onehot_cols = [col for col in y_train.columns if col.upper().startswith(name.upper() + "_")]
        y_dict[name] = y_train[onehot_cols]

    return y_dict


def process_predictions(pred_list: list[np.ndarray], output_info: list[dict[str, str]])-> dict[str, np.ndarray]:
    '''
    Convert prediction probabilities to class index per output.

    Example
    pred_list = [[0.1, 0.9], [0.7, 0.3]]
    output_info = ['name': 'GENDER']  
    -> {'GENDER': [1, 0]}
    '''
    pred_dict = {}

    for i, info in enumerate(output_info):
        name = info['name']
        pred = pred_list[i]

        pred_dict[name] = np.argmax(pred, axis=1)

    return pred_dict





