# Project-KKA-A Program Maps Ramah Disabilitas
---
Anggota:
| Nama | NRP |
| --- | --- | 
| Fauzan Hafiz Amandani  | 5025241087 |
| Nathanael oliver Amadhika  | 5025241109 |
| Raymond Julius Pardosi  | 5025241268 |
---

## **Deskripsi Proyek**

Program ini merupakan Sistem Pencarian Rute Multi-Modal Kota
Surabaya berbasis algoritma **A\* dengan penalti** dan **Greedy
Best-First Search**.\
Sistem ini mampu menghitung rute optimal antar lokasi dengan
mempertimbangkan moda transportasi berbeda seperti:

-   Bus
-   Trotoar / Pejalan Kaki
-   Ojek Online

Program juga mendukung aksesibilitas pengguna kursi roda, dengan
menghindari jalur yang tidak ramah disabilitas.

Proyek ini membantu pengguna menentukan rute perjalanan yang paling
efisien berdasarkan: <br>
- jarak tempuh, <br>
- penalti moda, <br>
- penalti transit, <br>
- penalti trotoar panjang (khusus kursi roda), <br>
- dan ketersediaan moda pada tiap edge.

------------------------------------------------------------------------

## **Program**

# 1. Representasi Graph

* Pemetaan Graph <br>
![WhatsApp Image 2025-10-26 at 01 53 51_9744e963](https://github.com/user-attachments/assets/b08d0f66-3c21-4f80-afe0-ef575a01741e)
<br>
* Jalur Bus <br>
![WhatsApp Image 2025-11-09 at 19 47 10_7dad0ab0](https://github.com/user-attachments/assets/23517bc1-b88e-46f9-a184-6140ef517972) <br>


Program menggunakan dua jenis node:

### • Node Grid

Node statis (A1--G2) yang mewakili titik pada grid peta.

### • Special Node

Lokasi khusus yang dihubungkan ke beberapa node grid, misalnya:

    "ITS": ["A1", "F4"]

Special node otomatis dibuatkan edge pendek (0.1 km) ke node-node
tersebut.

### • Edge Format

``` python
("A1", "A2", 1.000, {BUS, SIDEWALK}, True)
```

Edge menyimpan: - Node asal dan tujuan - Jarak (distance) - Moda
transport yang tersedia (set TransportMode) - Status aksesibilitas kursi
roda

Graph disimpan dalam dictionary bertingkat:

``` python
graph_map[node_a][node_b] = EdgeInfo(...)
```

------------------------------------------------------------------------

# 2. Sistem Moda Transport

Moda transportasi disimpan menggunakan `Enum`:

``` python
class TransportMode(Enum):
    BUS = "Bus"
    SIDEWALK = "Trotoar"
    OJEK = "Ojek Online"
```

Setiap edge memiliki himpunan moda:

``` python
modes: Set[TransportMode]
```

------------------------------------------------------------------------

# 3. Sistem Penalti

Kode menerapkan penalti pada biaya pergerakan:

### • Penalti Moda

``` python
PENALTIES = {
    BUS: 1.0,
    SIDEWALK: 1.5,
    OJEK: 3.5
}
```

### • Penalti Transit

Jika berpindah moda:

    TRANSIT_PENALTY = 0.5

### • Penalti Trotoar Berturut-turut

Untuk mendeteksi penggunaan sidewalk berturut-turut:

    CONSECUTIVE_SIDEWALK_PENALTY = 0.8

State yang dipertimbangkan pada A\*:

    (node, mode, consecutive_sidewalk_count)

------------------------------------------------------------------------

# 4. Heuristic (Euclidean Distance)

Perhitungan heuristik dihitung dengan:

``` python
heur[node] = sqrt((x - gx)^2 + (y - gy)^2)
```

Heuristik dipakai untuk: - A\* - Greedy Best-First Search

------------------------------------------------------------------------

# 5. PathNode

Setiap langkah rute disimpan dalam objek:

``` python
@dataclass
class PathNode:
    node: str
    mode: Optional[TransportMode]
    cumulative_cost: float
    actual_distance: float
```

`actual_distance` menyimpan jarak murni,\
`cumulative_cost` menyimpan jarak + penalti.

------------------------------------------------------------------------

# 6. A\* Multi-Modal

Fungsi utama:

``` python
multimodal_a_star(graph_map, start, goal, wheelchair_mode)
```

Modifikasi A\* dalam kode mencakup: - filtering jalur berdasarkan
aksesibilitas - penalti moda - penalti transit - penalti sidewalk
berturut-turut - state → `(node, mode, consecutive_sidewalk)` -
visited_costs untuk pruning

Priority queue menyimpan tuple:

    (f_cost, g_cost, actual_distance, path, current_mode, consec_sidewalk)

Output A\* berupa: - path (list PathNode) - total penalti - total jarak
aktual

------------------------------------------------------------------------

# 7. Greedy Best-First Search

Fungsi:

``` python
multimodal_greedy_bfs()
```

Ciri utama: - hanya menggunakan heuristik (tanpa cost) - tidak
menghitung penalti - lebih cepat tetapi tidak optimal

Queue diurutkan berdasarkan:

    heuristic(neighbor)

------------------------------------------------------------------------

# 8. Analisis Rute

Fungsi:

``` python
analyze_route(path)
```

Menghasilkan data statistik dari path: - total jarak per moda - jumlah
segmen per moda - jumlah transit - sidewalk berturut-turut terpanjang

Jarak segmen dihitung dengan:

``` python
path[i].actual_distance - path[i-1].actual_distance
```

------------------------------------------------------------------------

# 9. Output Formatting

Fungsi:

``` python
print_result(...)
```

Menampilkan: - urutan langkah node - moda per langkah - cost kumulatif -
statistik perjalanan

Hanya bagian presentasi, tidak mempengaruhi logika pencarian.

------------------------------------------------------------------------

# 10. Program Utama (CLI)

Menu CLI:

    1. Lihat daftar lokasi
    2. Lihat koneksi moda
    3. Cari rute (A*)
    4. Cari rute cepat (Greedy BFS)
    5. Keluar

`parse_choice()` digunakan untuk menerima input pengguna baik dalam
bentuk: - nomor indeks - nama node - special node

Jika start/goal merupakan special node, program mencoba semua opsi pintu
masuknya.

------------------------------------------------------------------------
