import os
import torch
import torch.nn.functional as F
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)
CORS(app)


#PATH MODEL
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR        = os.path.dirname(BASE_DIR)
PATH_TOKENIZER  = os.path.join(ROOT_DIR, "model_terbaik_80_10_10")
PATH_MODEL_PTH  = os.path.join(PATH_TOKENIZER, "model.pth")

HF_MODEL_NAME = "indolem/indobertweet-base-uncased"


#ARSITEKTUR MODEL
class SentimentIndoBERTweet(nn.Module):
    def __init__(self):
        super(SentimentIndoBERTweet, self).__init__()
        self.bert = AutoModel.from_pretrained(
            PATH_TOKENIZER,
            local_files_only=True
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier_sent = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        return self.classifier_sent(cls_output)

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = None
tokenizer = None


#MEMUAT MODEL
try:
    print(f"[INFO] Device      : {device}")
    print(f"[INFO] Tokenizer   : {PATH_TOKENIZER}")
    print(f"[INFO] Model file  : {PATH_MODEL_PTH}")

    #Cek file .pth
    if not os.path.exists(PATH_MODEL_PTH):
        raise FileNotFoundError(f"File model.pth tidak ditemukan di:\n   {PATH_MODEL_PTH}")

    #Load tokenizer dari folder lokal
    tokenizer = AutoTokenizer.from_pretrained(PATH_TOKENIZER)
    print("✅ Tokenizer berhasil dimuat")

    #Load arsitektur + bobot model
    model = SentimentIndoBERTweet()
    model.load_state_dict(
        torch.load(PATH_MODEL_PTH, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    print(f"✅ [BERHASIL] AI API Siap Digunakan! (Device: {device})")

except FileNotFoundError as e:
    print(f"[ERROR] File tidak ditemukan!\n {e}")
    model     = None
    tokenizer = None

except Exception as e:
    print(f"[ERROR] Gagal memuat model!")
    print(f"   Detail: {e}")
    model     = None
    tokenizer = None

SENT_CLASSES = ['Negatif', 'Positif']

#ENDPOINT DEBUG
@app.route('/debug', methods=['GET'])
def debug():
    return jsonify({
        'base_dir':          BASE_DIR,
        'path_tokenizer':    PATH_TOKENIZER,
        'path_model_pth':    PATH_MODEL_PTH,          
        'model_pth_exists':  os.path.exists(PATH_MODEL_PTH), 
        'tokenizer_exists':  os.path.exists(PATH_TOKENIZER),
        'model_loaded':      model is not None,
        'device':            str(device),
    })


#ENDPOINT 1: PREDIKSI TEKS TUNGGAL
@app.route('/predict', methods=['GET', 'POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if model is None or tokenizer is None:
        return jsonify({'error': 'Model belum dimuat. Cek log terminal server.'}), 503

    teks = request.json.get('text', '') if request.is_json else request.form.get('text', '')

    if not teks.strip():
        return jsonify({'error': 'Teks tidak boleh kosong.'}), 400

    try:
        inputs = tokenizer(
            teks, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits    = model(inputs['input_ids'], inputs['attention_mask'])
            prob_sent = F.softmax(logits, dim=1).cpu().numpy()[0]

        return jsonify({
            'status':   'success',
            'sentimen': SENT_CLASSES[prob_sent.argmax()],
            'detail_sentimen': {
                'Negatif': round(float(prob_sent[0]) * 100, 1),
                'Positif': round(float(prob_sent[1]) * 100, 1)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


#ENDPOINT 2: PREDIKSI FILE CSV
@app.route('/predict_csv', methods=['POST', 'OPTIONS'])
def predict_csv():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if model is None or tokenizer is None:
        return jsonify({'error': 'Model belum dimuat.'}), 503

    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'Tidak ada file yang dikirim.'}), 400

    try:
        df = pd.read_csv(request.files['file'])

        text_col, sent_col = None, None
        for col in df.columns:
            c = col.lower()
            if c in ['teks', 'text', 'ulasan', 'content', 'komentar']:
                text_col = col
            elif c in ['sentimen', 'sentiment', 'label_sentimen']:
                sent_col = col

        if not text_col:
            return jsonify({'error': 'CSV harus punya kolom "teks" atau "ulasan".'}), 400

        df             = df.dropna(subset=[text_col])
        teks_list      = df[text_col].astype(str).tolist()
        pred_sent_list = []

        for i in range(0, len(teks_list), 32):
            batch  = teks_list[i:i+32]
            inputs = tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                probs = F.softmax(
                    model(inputs['input_ids'], inputs['attention_mask']), dim=1
                ).cpu().numpy()
            for p in probs:
                pred_sent_list.append(SENT_CLASSES[p.argmax()])

        rekap = {
            'Negatif': pred_sent_list.count('Negatif'),
            'Positif': pred_sent_list.count('Positif')
        }

        result = {
            'status':           'success',
            'total':            len(teks_list),
            'rekap_sentimen':   rekap
        }

        if sent_col:
            true_sent = df[sent_col].astype(str).str.title().tolist()
            result['metrics_sentimen'] = {
                'accuracy':  round(accuracy_score(true_sent, pred_sent_list), 4),
                'precision': round(precision_score(true_sent, pred_sent_list, average='macro', zero_division=0), 4),
                'recall':    round(recall_score(true_sent, pred_sent_list, average='macro', zero_division=0), 4),
                'f1':        round(f1_score(true_sent, pred_sent_list, average='macro', zero_division=0), 4)
            }

        df['Prediksi_Sentimen'] = pred_sent_list
        result['csv_result']    = df.to_csv(index=False)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)