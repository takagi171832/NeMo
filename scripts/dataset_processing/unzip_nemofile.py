import os
import tarfile

def _unpack_nemo_file(path2file: str, out_folder: str, extract_config_only: bool = False) -> str:
    if not os.path.exists(path2file):
        raise FileNotFoundError(f"{path2file} does not exist")

    tar_header = "r:"
    try:
        tar_test = tarfile.open(path2file, tar_header)
        tar_test.close()
    except tarfile.ReadError:
        tar_header = "r:gz"
    tar = tarfile.open(path2file, tar_header)
    if not extract_config_only:
        tar.extractall(path=out_folder)
    else:
        members = [x for x in tar.getmembers() if ".yaml" in x.name]
        tar.extractall(path=out_folder, members=members)
    tar.close()
    return out_folder

if __name__ == "__main__":
    # ここに解凍したい .nemo ファイルのパスを設定してください
    path_to_nemo_file = "/home/takagi/NeMo/models/ASR/Librispeech/conformer_small/stt_en_conformer_ctc_small_ls.nemo"
    # ここに解凍したファイルを保存したいディレクトリのパスを設定してください
    output_folder = "/home/takagi/NeMo/models/ASR/Librispeech/conformer_small"
    # 必要に応じて、extract_config_only を True に設定して設定ファイルのみを抽出することができます
    extract_config_only = False

    # ファイルを解凍する
    _unpack_nemo_file(path_to_nemo_file, output_folder, extract_config_only)
