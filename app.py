from flask import Flask, render_template, request
import joblib

# Inisialisasi Flask
app = Flask(__name__)

# Muat model
model = joblib.load('model_kesehatan_hutan.pkl')

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data dari form
    kesehatan_pohon = int(request.form['kesehatan_pohon'])
    kondisi_tanah = int(request.form['kondisi_tanah'])
    kualitas_air = int(request.form['kualitas_air'])
    tingkat_gangguan = int(request.form['tingkat_gangguan'])

    # Buat prediksi menggunakan model
    prediksi = model.predict([[kesehatan_pohon, kondisi_tanah, kualitas_air, tingkat_gangguan]])

    # Kirim hasil ke halaman
    result = prediksi[0]
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
