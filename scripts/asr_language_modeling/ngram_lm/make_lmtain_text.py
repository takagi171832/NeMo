import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--manifest_file_path", type=str, default="/home/takagi/NeMo/csj/asr_ctc/manifests_S2A/SPS/train_sp/SPS_train_sp_manifest.json")
parser.add_argument("--output_file_path", type=str, default="/home/takagi/NeMo/csj/asr_ctc/lm_data/SPS_train.txt")
args = parser.parse_args()

manifest_file_path = args.manifest_file_path
output_file_path = args.output_file_path

# Process the manifest file and write the text into the output text file
with open(manifest_file_path, "r") as manifest_file, open(output_file_path, "w") as output_file:
    for line in manifest_file:
        entry = json.loads(line)
        text = entry["text"]
        # Insert spaces between characters
        spaced_text = " ".join(text)
        output_file.write(spaced_text + "\n")