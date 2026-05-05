from flask import Flask, render_template, request, redirect, session
import pandas as pd
import os

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

app = Flask(__name__)
app.secret_key = 'secret123'

# PATH
DATA_PATH = 'data/data_transaksi.xlsx'
OUTPUT_PATH = 'output/hasil_rules.csv'

# =========================
# LOGIN
# =========================
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form['username']
        pw = request.form['password']

        if user == 'admin' and pw == 'admin':
            session['login'] = True
            return redirect('/dashboard')
        else:
            return render_template('login.html', error="Login gagal")

    return render_template('login.html')


# =========================
# DASHBOARD
# =========================
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'login' not in session:
        return redirect('/')

    if request.method == 'POST':
        support = float(request.form['support'])
        confidence = float(request.form['confidence'])

        # cek file ada atau tidak
        if not os.path.exists(DATA_PATH):
            return "File data belum diupload!"

        df = pd.read_excel(DATA_PATH)

        # validasi kolom
        if 'Item' not in df.columns:
            return "Kolom 'Item' tidak ditemukan di Excel!"

        # bersihkan data
        df['Item'] = df['Item'].astype(str).str.strip()
        transaksi = df['Item'].apply(lambda x: [i.strip() for i in x.split(',')]).tolist()

        # encoding
        te = TransactionEncoder()
        te_ary = te.fit(transaksi).transform(transaksi)
        df_te = pd.DataFrame(te_ary, columns=te.columns_)

        # fp-growth
        freq = fpgrowth(df_te, min_support=support, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=confidence)

        # buat folder output kalau belum ada
        os.makedirs('output', exist_ok=True)
        os.makedirs('static', exist_ok=True)

        # simpan hasil
        rules.to_csv(OUTPUT_PATH, index=False)

        # grafik
        import matplotlib.pyplot as plt
        freq.sort_values(by='support', ascending=False).head(10).plot(
            x='itemsets', y='support', kind='bar', legend=False
        )
        plt.tight_layout()
        plt.savefig('static/grafik.png')
        plt.close()

        return redirect('/hasil')

    return render_template('dashboard.html')


# =========================
# UPLOAD
# =========================
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'login' not in session:
        return redirect('/')

    if request.method == 'POST':
        file = request.files['file']

        if file.filename == '':
            return "Tidak ada file dipilih!"

        os.makedirs('data', exist_ok=True)
        file.save(DATA_PATH)

        return redirect('/dashboard')

    return render_template('upload.html')


# =========================
# HASIL
# =========================
@app.route('/hasil')
def hasil():
    if 'login' not in session:
        return redirect('/')

    if not os.path.exists(OUTPUT_PATH):
        return "Belum ada hasil"

    df = pd.read_csv(OUTPUT_PATH)

    return render_template('hasil.html',
                           columns=df.columns,
                           data=df.values,
                           grafik=True)


# =========================
# LOGOUT
# =========================
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')


# =========================
# RUN (LOKAL + RENDER)
# =========================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)