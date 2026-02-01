# Flocking Simulation – CPU (Naive, OpenMP) i GPU (CUDA)

## Opis projekta

Ovaj projekat implementira simulaciju **flocking** ponašanja (Boids model), gde se veliki broj jedinki (boida) kreće u prostoru prateći tri osnovna pravila:

* **Alignment** – usklađivanje pravca kretanja sa susedima
* **Cohesion** – kretanje ka centru mase suseda
* **Separation** – izbegavanje sudara sa previše bliskim jedinkama

Cilj projekta je **poređenje performansi** različitih implementacija:

* Naivna sekvencijalna CPU verzija
* Paralelizovana CPU verzija pomoću **OpenMP**
* Naivna **GPU (CUDA)** verzija
* Optimizovana **GPU (CUDA)** verzija sa prostornim segmentiranjem (uniform grid)

Simulacija se izvršava u realnom vremenu, stoga renderovanje predstavlja dodatno usko grlo pri merenju ukupnih performansi.

---

## Ulazni parametri

Program prihvata sledeće parametre (putem komandne linije ili podrazumevanih vrednosti):

* **N** – broj boida (jedinki)
* **perception_radius** – radijus u kojem boid traži susede
* **fov_deg** – ugao vidnog polja (Field of View)
* **w_align** – težinski faktor za alignment
* **w_cohesion** – težinski faktor za cohesion
* **w_separation** – težinski faktor za separation

Primer (ako se koristi komandna linija):

```bash
./jato N perception_radius fov_deg w_align w_cohesion w_separation mode
```

---

## Pokretanje programa

### Kompajliranje

Projekat koristi:

* **gcc** za C kod
* **nvcc** za CUDA deo
* **SDL2** za renderovanje

Kompajliranje se vrši pomoću `Makefile` fajla:

```bash
make
```

Nakon uspešne kompilacije dobija se izvršni fajl:

```bash
jato.exe
```

---

### Pokretanje simulacije

```bash
./jato.exe 1000 50 270 1.0 0.8 1.2
```

---

## Kontrole unutar programa

Tokom izvršavanja simulacije, korisnik može **dinamički menjati parametre** pomoću tastature:

Podešavanje parametara ponašanja boida
| Taster | Akcija                                |
| ------ | ------------------------------------- |
| **W**  | Povećava *alignment* težinski faktor  |
| **S**  | Smanjuje *alignment* težinski faktor  |
| **A**  | Povećava *cohesion* težinski faktor   |
| **D**  | Smanjuje *cohesion* težinski faktor   |
| **Z**  | Povećava *separation* težinski faktor |
| **X**  | Smanjuje *separation* težinski faktor |
| **Q**  | Povećava ugao vidnog polja (*FOV*)    |
| **E**  | Smanjuje ugao vidnog polja (*FOV*)    |
| **R**  | Povećava radijus percepcije           |
| **F**  | Smanjuje radijus percepcije           |

Vizuelne opcije
| Taster | Akcija                                           |
| ------ | ------------------------------------------------ |
| **C**  | Uključuje / isključuje prikaz vidnog polja (FOV) |

Promena režima izvršavanja
| Taster | Režim                                                |
| ------ | ---------------------------------------------------- |
| **T**  | Promena izvrsavanja (CPU) i (GPU)                    |
| **Y**  | OpenMP paralelizacija (CPU)                          |
| **U**  | CUDA GPU implementacija sa prostornim segmentiranjem |

Ostalo
| Taster  | Akcija              |
| ------- | ------------------- |
| **ESC** | Izlazak iz programa |


---

## Napomena o performansama

* Naivna CPU verzija ima kvadratnu složenost **O(N²)** i brzo postaje neupotrebljiva za veći broj boida.
* OpenMP verzija pokazuje značajno ubrzanje uz minimalne izmene koda.
* GPU verzije ostvaruju najveći FPS, naročito optimizovana varijanta koja koristi **uniform grid** i razmatra samo lokalne susede.
* Renderovanje predstavlja **usko grlo**, pa FPS ne odražava isključivo računsku snagu GPU-a.

---

## Kratak rezime

Projekat demonstrira kako se isti problem može rešavati različitim paralelnim pristupima i kako pravilna organizacija podataka i memorijskog pristupa na GPU-u (lokalnost, sortiranje, grid strukture) može dramatično poboljšati performanse.
