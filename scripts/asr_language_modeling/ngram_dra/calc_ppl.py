import kenlm
import json

def calculate_ppl(model_path, sentences):
    # 言語モデルをロード
    model = kenlm.Model(model_path)

    # 各文のPPLを計算
    total_ppl = 0
    for sentence in sentences:
        # パープレキシティの計算
        score = model.score(sentence)
        ppl = 10**(-score / len(sentence.split()))
        total_ppl += ppl

    # 平均パープレキシティの算出
    average_ppl = total_ppl / len(sentences)
    return average_ppl

def main():
    # 言語モデルのフォルダパス
    model_path = "/home/takagi/NeMo/models/LM"
    #　テストデータのファイルパスのリスト
    test_file_path = ["/home/takagi/NeMo/manifests/laboroTV/dev_4k/dev_4k_manifest.json", 
                      "/home/takagi/NeMo/manifests/laboroTV/dev/dev_manifest.json",
                      "/home/takagi/NeMo/manifests/laboroTV/tedx-jp-10k/tedx-jp-10k_manifest.json",
                      "/home/takagi/NeMo/manifests/CSJ/eval1/eval1_manifest.json",
                      "/home/takagi/NeMo/manifests/CSJ/eval2/eval2_manifest.json",
                      "/home/takagi/NeMo/manifests/CSJ/eval3/eval3_manifest.json",]
    # データセットのリスト
    datasets = ["laboro", "csj"]

    # 結果を保存するファイルパス
    output_file_path = "/home/takagi/NeMo/dataset/output/ppl/perplexity.txt"

    # テストデータの読み込み
    dev_4k_sentences = []
    dev_sentences = []
    tedx_sentences = []
    csj_eval1_sentences = []
    csj_eval2_sentences = []
    csj_eval3_sentences = []

    with open(test_file_path[0], "r") as f:
        for line in f:
            entry = json.loads(line)
            text = entry["text"]
            spaced_text = " ".join(text)
            dev_4k_sentences.append(spaced_text)
    
    with open(test_file_path[1], "r") as f:
        for line in f:
            entry = json.loads(line)
            text = entry["text"]
            spaced_text = " ".join(text)
            dev_sentences.append(spaced_text)
    
    with open(test_file_path[2], "r") as f:
        for line in f:
            entry = json.loads(line)
            text = entry["text"]
            spaced_text = " ".join(text)
            tedx_sentences.append(spaced_text)

    with open(test_file_path[3], "r") as f:
        for line in f:
            entry = json.loads(line)
            text = entry["text"]
            spaced_text = " ".join(text)
            csj_eval1_sentences.append(spaced_text)
    
    with open(test_file_path[4], "r") as f:
        for line in f:
            entry = json.loads(line)
            text = entry["text"]
            spaced_text = " ".join(text)
            csj_eval2_sentences.append(spaced_text)
    
    with open(test_file_path[5], "r") as f:
        for line in f:
            entry = json.loads(line)
            text = entry["text"]
            spaced_text = " ".join(text)
            csj_eval3_sentences.append(spaced_text)

    # パープレキシティの計算
    for dataset in datasets:
        # ngramオーダーが２~６でループ
        for n in range(2, 7):
            # 言語モデルのファイルパス
            lm_file_path = model_path + "/" + dataset + str(n) + "gram.bin"
            
            # パープレキシティの計算
            laboro_dev_4k_ppl = calculate_ppl(lm_file_path, dev_4k_sentences)
            laboro_dev_ppl = calculate_ppl(lm_file_path, dev_sentences)
            laboro_tedx_ppl = calculate_ppl(lm_file_path, tedx_sentences)
            csj_eval1_ppl = calculate_ppl(lm_file_path, csj_eval1_sentences)
            csj_eval2_ppl = calculate_ppl(lm_file_path, csj_eval2_sentences)
            csj_eval3_ppl = calculate_ppl(lm_file_path, csj_eval3_sentences)
            
            # 結果の保存
            with open(output_file_path, "a") as f:
                f.write("lm: " + lm_file_path + "\n")
                f.write("ngramオーダーが" + str(n) + "の場合\n")
                f.write("laboro_dev_4k_ppl: " + str(laboro_dev_4k_ppl) + "\n")
                f.write("laboro_dev_ppl: " + str(laboro_dev_ppl) + "\n")
                f.write("laboro_tedx_ppl: " + str(laboro_tedx_ppl) + "\n")
                f.write("csj_eval1_ppl: " + str(csj_eval1_ppl) + "\n")
                f.write("csj_eval2_ppl: " + str(csj_eval2_ppl) + "\n")
                f.write("csj_eval3_ppl: " + str(csj_eval3_ppl) + "\n")
                f.write("\n")

if __name__ == "__main__":
    main()