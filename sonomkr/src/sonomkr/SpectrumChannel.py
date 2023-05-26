import json

G10 = 10.0 ** (3.0/10.0)
G2 = 2.0
BOCT = 1.0
BTHIRD = 3.0
R48 = 48000
R44P1 = 44100

# strategy : base 2 or base 10
G = G10
# Nth octave band : 1 or 3
B = BTHIRD
# Sampling rate
R = R44P1
NQST = R / 2.0
# Filters order
ORDER = 6

f1000 = 1000.0
frequencies = {}

min_band_center_freq = 0

max_band_number = 13 if B == BTHIRD else 3
min_band_number = -31 if B == BTHIRD else -10
band_numbers = range(min_band_number, max_band_number+1)

for x in band_numbers:
    fmid = (G ** (x / B)) * f1000
    fmax = (G ** (1 / (2 * B))) * fmid
    fmin = (G ** (- 1 / (2 * B))) * fmid
    frequencies[x] = {
        "center": fmid,
        "max": fmax,
        "min": fmin
    }
print(json.dumps(frequencies, sort_keys=True, indent=4))