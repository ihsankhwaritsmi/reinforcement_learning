# Laporan Proyek Reinforcement Learning (DQN untuk Ms. Pac-Man)

Proyek ini mengimplementasikan agen Deep Q-Network (DQN) untuk bermain game Ms. Pac-Man menggunakan lingkungan Gymnasium dan TensorFlow.

## I. Pengaturan Lingkungan

Lingkungan permainan Ms. Pac-Man dibuat menggunakan pustaka Gymnasium, penerus OpenAI's Gym, yang berinteraksi dengan Arcade Learning Environment (ALE). Beberapa "wrapper" diterapkan untuk memproses awal bingkai permainan, membuatnya lebih cocok untuk pelatihan jaringan saraf.

Fungsi `make_env` bertanggung jawab untuk:
- Membuat lingkungan dasar Ms. Pac-Man.
- Mengubah ukuran observasi menjadi 84x84 piksel.
- Mengubah observasi menjadi skala abu-abu.
- Menumpuk 4 bingkai berturut-turut untuk menangkap informasi temporal (gerakan).

Berikut adalah contoh lingkungan Ms. Pac-Man:

![Lingkungan Ms. Pac-Man](images/pacman_environment.png)

## II. Model Deep Q-Network (DQN)

Agen menggunakan Deep Q-Network (DQN) untuk mempelajari kebijakan optimal. DQN adalah jaringan saraf tiruan konvolusional (CNN) yang mengambil bingkai permainan yang sudah diproses sebagai masukan dan menghasilkan nilai Q untuk setiap tindakan yang mungkin.

Arsitektur model DQN terdiri dari:
- Lapisan masukan yang menerima tumpukan 4 bingkai (84x84 piksel).
- Lapisan `Permute` untuk mengubah dimensi masukan agar sesuai dengan format `channels_last` yang diharapkan oleh TensorFlow.
- Tiga lapisan konvolusional (`Conv2D`) dengan fungsi aktivasi ReLU untuk mengekstraksi fitur dari bingkai permainan.
- Lapisan `Flatten` untuk meratakan keluaran konvolusional.
- Dua lapisan `Dense` (terhubung penuh) dengan fungsi aktivasi ReLU dan linear untuk menghasilkan nilai Q untuk setiap tindakan.

## III. Buffer Pengalaman (Replay Buffer)

Untuk menstabilkan pelatihan, sebuah buffer pengalaman digunakan untuk menyimpan pengalaman agen (state, action, reward, next_state, done). Agen kemudian mengambil sampel acak dari buffer ini untuk melatih model DQN. Ini membantu memutus korelasi antara pengalaman berturut-turut dan meningkatkan stabilitas proses pelatihan.

## IV. Agen DQN

Kelas `DQNAgent` merangkum perilaku agen, termasuk:
- **Model DQN utama**: Digunakan untuk pemilihan tindakan dan pelatihan.
- **Model target**: Salinan model utama yang diperbarui secara berkala untuk menghitung nilai Q target, membantu menstabilkan pelatihan.
- **Buffer pengalaman**: Untuk menyimpan dan mengambil sampel pengalaman.
- **Kebijakan Epsilon-Greedy**: Untuk menyeimbangkan eksplorasi (mencoba tindakan baru) dan eksploitasi (mengambil tindakan terbaik yang diketahui). Nilai epsilon berkurang seiring waktu.
- **Optimizer Adam** dan **fungsi kerugian Huber**: Digunakan untuk melatih model.

## V. Loop Pelatihan

Loop pelatihan adalah inti dari proses pembelajaran. Dalam setiap episode:
1. Agen berinteraksi dengan lingkungan, memilih tindakan berdasarkan kebijakan epsilon-greedy.
2. Pengalaman (state, action, reward, next_state, done) disimpan dalam buffer pengalaman.
3. Agen melatih model DQN menggunakan sampel dari buffer pengalaman.
4. Model target diperbarui secara berkala untuk menstabilkan pelatihan.

Berikut adalah contoh hasil pelatihan per episode:

![Hasil Pembelajaran per Episode](images/hasil_learning_episode.png)

## Kesimpulan

Proyek ini berhasil mengimplementasikan agen DQN untuk bermain Ms. Pac-Man. Dengan menggunakan teknik seperti preprocessing bingkai, replay buffer, dan model target, agen dapat belajar kebijakan yang efektif untuk memaksimalkan hadiah dalam permainan.
